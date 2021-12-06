import json
import os
import sys
import torch
import tensorflow as tf
from tensorflow.keras import layers
import re
import argparse
import subprocess

sys.path.insert(0, f'{os.path.dirname(sys.argv[0])}/..')
from utils import export_onnx_fix_batch, freeze_graph

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
        self.conv0 = torch.nn.Conv2d(3, cin, 1, padding='same')
        self.conv1 = torch.nn.Conv2d(cin, cout, k, padding='same')

    def forward(self, x):
        return self.conv1(self.conv0(x))    


def get_dummy_dwconv(cin, mul, k, hw):
    return tf.keras.Sequential([
        tf.keras.Input(shape=[hw, hw, 3], batch_size=1),
        layers.Conv2D(filters=cin, kernel_size=1, padding='same'),
        layers.DepthwiseConv2D(kernel_size=k, padding='same', depth_multiplier=mul)
    ])

def get_dummy_dense(batch_size, in_dim, out_dim):
    return tf.keras.Sequential([
        tf.keras.Input(shape=[1, 1, 3], batch_size=batch_size),
        layers.Conv2D(filters=in_dim, kernel_size=1, padding='same'),
        layers.Reshape([batch_size, in_dim]),
        layers.Dense(out_dim)
    ])

class OpTester:
    def __init__(self, vino_model_dir: str, dataset_dir: str):
        self.src_model_dir = os.path.join(vino_model_dir, 'src_model', 'quant_op_test')
        self.ir_model_dir = os.path.join(vino_model_dir, 'ir', 'quant_op_test')
        self.dataset_dir = dataset_dir
        self.model_zoo = {
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
            },
            # dwconv
            'dwconv_kernel': {
                'name': 'dwconv_k{:01}_io32_hw56',
                'range': [1, 3, 5, 7],
                'param': [32, 1, None, 56], # [cin, depth_multiplier, kernel_size, hw]
            },
            'dwconv_kernel2': {
                'name': 'dwconv_k{:01}_io320_hw14',
                'range': [1, 3, 5, 7],
                'param': [320, 1, None, 14],
            },
            'dwconv_cio': {
                'name': 'dwconv_k3_io{:03}_hw56',
                'range': range(1, 129),
                'param': [None, 1, 3, 56]
            },
            # dense 
            'dense2d_out': {
                'name': 'dense2d1_192_{:03}',
                'range': range(1, 129),
                'param': [1, 192, None] # [batch_size=1, input_size, output_size]
            },
            'dense2d_in': {
                'name': 'dense2d1_{:03}_192',
                'range': range(1, 129),
                'param': [1, None, 192]
            },
            'dense2d_out768': {
                'name': 'dense2d1_768_{:04}',
                'range': [1<<x for x in range(13)],
                'param': [1, 768, None]
            },
            'dense2d_in768': {
                'name': 'dense2d1_{:04}_768',
                'range': [1<<x for x in range(13)],
                'param': [1, None, 768]
            },
            'dense2d_out384': {
                'name': 'dense2d1_384_{:04}',
                'range': [1<<x for x in range(13)],
                'param': [1, 384, None]
            },
            'dense2d_in384': {
                'name': 'dense2d1_{:04}_384',
                'range': [1<<x for x in range(13)],
                'param': [1, None, 384]
            },
            'dense2d_out192': {
                'name': 'dense2d1_192_{:04}',
                'range': [1<<x for x in range(13)],
                'param': [1, 192, None]
            },
            'dense2d_in192': {
                'name': 'dense2d1_{:04}_192',
                'range': [1<<x for x in range(13)],
                'param': [1, None, 192]
            },
        }

    def _get_src_onnx_path(self, k, v, i) -> str:
        return os.path.join(self.src_model_dir, k, v['name'].format(i)+'.onnx')

    def _get_src_pb_path(self, k, v, i) -> str:
        return os.path.join(self.src_model_dir, k, v['name'].format(i)+'.pb')

    def _get_src_path(self, k, v, i) -> str:
        if re.match('conv', v['name']):
            return self._get_src_onnx_path(k, v, i)
        elif re.match('dwconv', v['name']):
            return self._get_src_pb_path(k, v, i)
        elif re.match('dense2d', v['name']):
            return self._get_src_pb_path(k, v, i)
        else:
            raise ValueError('Model config not supported')

    def _get_fp32_ir_dir(self, k, v, i) -> str:
        return os.path.join(self.ir_model_dir, k, v['name'].format(i), 'FP32')

    def _export_src(self, ):
        print('===== Exporting SRC =====')
        for test_case, v in self.model_zoo.items():
            for i in v['range']:
                output_path = self._get_src_path(test_case, v, i)
                param = [x or i for x in v['param']]

                if re.match('conv', v['name']):
                    model = DummyConv(*param[:3])
                    export_onnx_fix_batch(model, output_path, input_shape=[1, 3, param[3], param[3]])
                elif re.match('dwconv', v['name']):
                    model = get_dummy_dwconv(*param)
                    freeze_graph(keras_model=model, output_path=output_path)
                elif re.match('dense2d', v['name']):
                    model = get_dummy_dense(*param)
                    freeze_graph(keras_model=model, output_path=output_path)
                else:
                    raise RuntimeError()

    def _convert(self, ):
        print('===== Converting SRC to OpenVINO IR =====')
        for test_case, v in self.model_zoo.items():
            for i in v['range']:
                model_name = v['name'].format(i)
                input_model = self._get_src_path(test_case, v, i)
                output_dir = self._get_fp32_ir_dir(test_case, v, i)
                subprocess.run(f'python $VINO_MO --input_model={input_model} --model_name={model_name} --output_dir={output_dir} --batch=1 --data_type=FP32', shell=True)

    def _quantize(self, ):
        print('==== Quantizing FP32IR to INT8 ====')
        for test_case, v in self.model_zoo.items():
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

    def _get_profiling_pattern(self, test_case):
        patten_dict = {
            'conv': (r'.*Conv_1.*', None),
            'dwconv': (r'.*depthwise_conv2d.*', r'GroupConvolution'),
            'dense': (r'.*dense.*', None)
        }
        for k, v in patten_dict.items():
            if re.match(k, test_case):
                return v 
        raise RuntimeError()

    def _benchmark(self):
        print('====== Benchmarking model performance on CPU with 1 thread ======')
        result_summary = {}
        for test_case, v in self.model_zoo.items():
            print(f'==    Testing {test_case}    ==')
            result_summary[test_case] = {}
            latency_list_fp32 = []
            latency_list_int8 = []
            layername_pattern, layertype_pattern = self._get_profiling_pattern(test_case)
            for i in v['range']:
                model_name = v['name'].format(i)
                xml_path_fp32 = os.path.join(self._get_fp32_ir_dir(test_case, v, i), model_name+'.xml')
                xml_path_int8 = xml_path_fp32.replace('/FP32/', '/FP32-INT8/')
  
                latency_fp32 = openvino_benchmark('$VINO_BENCHMARK_APP', xml_path_fp32, niter=30, num_threads=1, batch_size=1, layername_pattern=layername_pattern, layertype_pattern=layertype_pattern, csv_output_dir=os.path.dirname(xml_path_fp32), show_detail=False)
                latency_int8 = openvino_benchmark('$VINO_BENCHMARK_APP', xml_path_int8, niter=30, num_threads=1, batch_size=1, layername_pattern=layername_pattern, layertype_pattern=layertype_pattern, csv_output_dir=os.path.dirname(xml_path_int8), show_detail=False)

                latency_list_fp32.append(latency_fp32)
                latency_list_int8.append(latency_int8)
            print(f'--- {test_case} Summary ---')
            print('FP32 ms')
            print([round(x, 2) for x in latency_list_fp32])
            print('INT8 ms')
            print([round(x, 2) for x in latency_list_int8])
            result_summary[test_case]['fp32'] = [round(x, 2) for x in latency_list_fp32]
            result_summary[test_case]['int8'] = [round(x, 2) for x in latency_list_int8]

        print('==================================================================================')
        print('                                  SUMMARY')
        print('==================================================================================')

        for test_case, v in result_summary.items():
            print(f'--- {test_case} ---')
            print('FP32 ms')
            print(v['fp32'])
            print('INT8 ms')
            print(v['int8'])
            print('')
            print('')


    def run(self, ):
        self._export_src()
        self._convert()
        self._quantize()
        self._benchmark()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vino_model_dir', required=True, type=str, help='vino model root dir to save SRC(tf, onnx) and IR model')
    parser.add_argument('--dataset_dir', required=True, type=str, help='dataset dir to do quantization')
    args = parser.parse_args()

    op_tester = OpTester(args.vino_model_dir, args.dataset_dir)
    op_tester.run()


if __name__ == '__main__':
    main()