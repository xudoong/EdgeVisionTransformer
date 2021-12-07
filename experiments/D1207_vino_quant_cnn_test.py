from tensorflow import keras
from modeling.models.cnn_zoo import cnn_zoo_dict
import os
import sys
import subprocess
import argparse
import re
import json

sys.path.insert(0, f'{os.path.dirname(sys.argv[0])}/..')
from utils import freeze_graph

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

class VinoCnnTester:
    def __init__(self, cnn_zoo_dict: dict, vino_model_dir: str, dataset_dir: str):
        self.src_model_dir = os.path.join(vino_model_dir, 'src_model', 'quant_cnn_test')
        self.ir_model_dir = os.path.join(vino_model_dir, 'ir', 'quant_cnn_test')
        self.dataset_dir = dataset_dir
        self.cnn_zoo_dict = cnn_zoo_dict

    def _get_src_pb_path(self, model_name) -> str:
        return os.path.join(self.src_model_dir, model_name+'.pb')

    def _get_fp32_ir_dir(self, model_name) -> str:
        return os.path.join(self.ir_model_dir, model_name, 'FP32')

    def _export_src(self, ):
        print('===== Exporting SRC =====')
        for model_name, generator_func in self.cnn_zoo_dict.items():
            model = generator_func()
            freeze_graph(keras_model=model, output_path=self._get_src_pb_path(model_name))

    def _convert(self, ):
        print('===== Converting SRC to OpenVINO IR =====')
        for model_name in self.cnn_zoo_dict.keys():
            input_path = self._get_src_pb_path(model_name)
            output_dir = self._get_fp32_ir_dir(model_name)
            subprocess.run(f'python $VINO_MO --input_model={input_path} --model_name={model_name} --output_dir={output_dir} --batch=1 --data_type=FP32', shell=True)

    def _quantize(self, ):
        print('==== Quantizing FP32IR to INT8 ====')
        for model_name in self.cnn_zoo_dict.keys():
            fp32_ir_dir = self._get_fp32_ir_dir(model_name)
            output_dir = fp32_ir_dir.replace('/FP32', '/FP32-INT8')
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
        latency_list_fp32 = []
        latency_list_int8 = []
        for model_name in self.cnn_zoo_dict.keys():
            xml_path_fp32 = os.path.join(self._get_fp32_ir_dir(model_name), model_name+'.xml')
            xml_path_int8 = xml_path_fp32.replace('/FP32/', '/FP32-INT8/')
            latency_fp32 = 0.0
            latency_int8 = 0.0
            try:
                latency_fp32 = openvino_benchmark('$VINO_BENCHMARK_APP', xml_path_fp32, niter=30, num_threads=1, batch_size=1, csv_output_dir=os.path.dirname(xml_path_fp32), show_detail=False)
                latency_int8 = openvino_benchmark('$VINO_BENCHMARK_APP', xml_path_int8, niter=30, num_threads=1, batch_size=1, csv_output_dir=os.path.dirname(xml_path_int8), show_detail=False)
            except:
                pass
            latency_list_fp32.append(round(latency_fp32, 2))
            latency_list_int8.append(round(latency_int8, 2))

        print('==================================================================================')
        print('                                  SUMMARY')
        print('==================================================================================')

        print(list(self.cnn_zoo_dict.keys()))
        print('FP32 ms')
        print(latency_list_fp32)
        print('INT8 ms')
        print(latency_list_int8)

    def run(self, ):
        # self._export_src()
        # self._convert()
        # self._quantize()
        self._benchmark()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vino_model_dir', required=True, type=str, help='vino model root dir to save SRC(tf, onnx) and IR model')
    parser.add_argument('--dataset_dir', required=True, type=str, help='dataset dir to do quantization')
    args = parser.parse_args()

    vino_cnn_tester = VinoCnnTester(cnn_zoo_dict, args.vino_model_dir, args.dataset_dir)
    vino_cnn_tester.run()


if __name__ == '__main__':
    main()