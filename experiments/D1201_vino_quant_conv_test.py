import json
import os
import sys
import torch
import argparse
import subprocess

sys.path.insert(0, f'{os.path.dirname(sys.argv[0])}/..')
import utils

sys.path.insert(0, f'{os.path.dirname(sys.argv[0])}/..')
from benchmark.openvino.vino_cli import openvino_benchmark

class PotConfigJson:
    def __init__(self, model_xml_path: str, dataset_path) -> None:
        model_name = os.path.splitext(os.path.basename(model_xml_path))[0]
        self.pot_config = {
            "model": {
                "model_name": f"{model_name}",
                "model": f"{model_xml_path}",
                "weights": f"{model_xml_path.replace('.xml', '.bin')}"
            },
            "engine": {
                "type": "simplified",
                "data_source": f"{dataset_path}"
            },
            "compression": {
                "target_device": "CPU",
                "algorithms": [
                    {
                        "name": "DefaultQuantization",
                        "params": {
                            "preset": "performance",
                            "stat_subset_size": 3
                        }
                    }
                ]
            }
        }   

    def dump(self, output_path):
        with open(output_path, 'w') as f:
            json.dump(self.pot_config, f, indent=4)
            f.write('\n')


class DummyConv(torch.nn.Module):
    def __init__(self, cin, cout, k):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, cin, 1, padding=0)
        self.conv1 = torch.nn.Conv2d(cin, cout, k, padding=(k-1)//2)

    def forward(self, x):
        return self.conv1(self.conv0(x))    


class ConvExporter:
    def __init__(self, vino_model_dir: str, dataset_dir: str):
        self.onnx_model_dir = os.path.join(vino_model_dir, 'onnx_model', 'quant_op_test')
        self.ir_model_dir = os.path.join(vino_model_dir, 'ir', 'quant_op_test')
        self.dataset_dir = dataset_dir
        self.conv_zoo = {
            'conv_kernel': {
                'name': 'conv_k{}_i32_o32_hw56',
                'range': [1, 3, 5, 7],
                'param': [32, 32, None, 56], # [cin, cout, kernel_size, hw]
            },
            'conv_kernel2': {
                'name': 'conv_k{}_i320_o320_hw14',
                'range': [1, 3, 5, 7],
                'param': [320, 320, None, 14],
            },
            'conv_cout': {
                'name': 'conv_k3_i32_o{:03}_hw56',
                'range': range(1, 129),
                'param': [32, None, 3, 56],
            },
            'conv_cin': {
                'name': 'conv_k3_i{:03}_o32_hw56',
                'range': range(1, 129),
                'param': [None, 32, 3, 56],
            }
        }

    def _get_onnx_path(self, k, v, i) -> str:
        return os.path.join(self.onnx_model_dir, k, v['name'].format(i)+'.onnx')

    def _get_fp32_ir_dir(self, k, v, i) -> str:
        return os.path.join(self.ir_model_dir, k, v['name'].format(i), 'FP32')

    def _export_onnx(self, ):
        print('===== Exporting ONNX =====')
        for test_case, v in self.conv_zoo.items():
            for i in v['range']:
                output_path = self._get_onnx_path(test_case, v, i)
                model = DummyConv(*[(v['param'][_] or i) for _ in range(3)])
                utils.export_onnx_fix_batch(model, output_path, input_shape=[1, 3, v['param'][3], v['param'][3]])

    def _convert(self, ):
        print('===== Converting ONNX to OpenVINO IR =====')
        for test_case, v in self.conv_zoo.items():
            for i in v['range']:
                model_name = v['name'].format(i)
                input_model = self._get_onnx_path(test_case, v, i)
                output_dir = self._get_fp32_ir_dir(test_case, v, i)
                subprocess.run(f'python $VINO_MO --input_model={input_model} --model_name={model_name} --output_dir={output_dir} --batch=1 --data_type=FP32', shell=True)

    def _quantize(self, ):
        print('==== Quantizing FP32IR to INT8 ====')
        for test_case, v in self.conv_zoo.items():
            for i in v['range']:
                fp32_ir_dir = self._get_fp32_ir_dir(test_case, v, i)
                output_dir = fp32_ir_dir.replace('/FP32', '/FP32-INT8')
                model_name = v['name'].format(i)
                if os.path.exists(output_dir):
                    subprocess.run(f'rm -r {output_dir}', shell=True)
                os.mkdir(output_dir)
                # save pot-config.json
                pot_config_json_path = os.path.join(output_dir, 'pot-config.json')
                pot_config = PotConfigJson(
                    model_xml_path=os.path.join(fp32_ir_dir, model_name+'.xml'),
                    dataset_path=self.dataset_dir
                )
                pot_config.dump(pot_config_json_path)
                # quant
                subprocess.run(f'python -m pot -c {pot_config_json_path} --direct-dump --output-dir={output_dir}', shell=True)
                # move file
                subprocess.run(f'mv {output_dir}/optimized/{model_name}.xml {output_dir}', shell=True)
                subprocess.run(f'mv {output_dir}/optimized/{model_name}.bin {output_dir}', shell=True)

    def _benchmark(self):
        print('====== Benchmarking model performance on CPU with 1 thread ======')
        result_summary = {}
        for test_case, v in self.conv_zoo.items():
            print(f'==    Testing {test_case}    ==')
            result_summary[test_case] = {}
            latency_list_fp32 = []
            latency_list_int8 = []
            for i in v['range']:
                model_name = v['name'].format(i)
                xml_path_fp32 = os.path.join(self._get_fp32_ir_dir(test_case, v, i), model_name+'.xml')
                xml_path_int8 = xml_path_fp32.replace('/FP32/', '/FP32-INT8/')
  
                latency_fp32 = openvino_benchmark('$VINO_BENCHMARK_APP', xml_path_fp32, niter=30, num_threads=1, batch_size=1, re_pattern=r'.*Conv_1.*', csv_output_dir=os.path.dirname(xml_path_fp32), show_detail=False)
                latency_int8 = openvino_benchmark('$VINO_BENCHMARK_APP', xml_path_int8, niter=30, num_threads=1, batch_size=1, re_pattern=r'.*Conv_1.*', csv_output_dir=os.path.dirname(xml_path_int8), show_detail=False)

                latency_list_fp32.append(latency_fp32)
                latency_list_int8.append(latency_int8)
            print(f'--- {test_case} Summary ---')
            print('FP32 ms')
            print([round(x, 2) for x in latency_list_fp32])
            print('INT8 ms')
            print([round(x, 2) for x in latency_list_int8])
            result_summary[test_case]['fp32'] = [round(x, 2) for x in latency_list_fp32]
            result_summary[test_case]['int8'] = [round(x, 2) for x in latency_list_int8]

        print('== SUMMARY ==')
        for test_case, v in result_summary.items():
            print(f'--- {test_case} ---')
            print('FP32 ms')
            print(v['fp32'])
            print('INT8 ms')
            print(v['int8'])

    def run(self, ):
        self._export_onnx()
        self._convert()
        self._quantize()
        self._benchmark()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vino_model_dir', required=True, type=str, help='vino model root dir to save onnx and IR model')
    parser.add_argument('--dataset_dir', required=True, type=str, help='dataset dir to do quantization')
    args = parser.parse_args()

    conv_exporter = ConvExporter(args.vino_model_dir, args.dataset_dir)
    conv_exporter.run()


if __name__ == '__main__':
    main()