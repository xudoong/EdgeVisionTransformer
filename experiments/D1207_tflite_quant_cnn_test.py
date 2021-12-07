import argparse
import re
import os
import sys
import subprocess
import re

sys.path.insert(0, f'{os.path.dirname(sys.argv[0])}/..')
from modeling.models.cnn_zoo import cnn_zoo_dict
from utils import tf2tflite_dir

class ADB:
    def __init__(self, serino):
        self.serino = serino
    
    def push(self, src, dst):
        subprocess.run(f'adb -s {self.serino} push {src} {dst}', shell=True)

    def pull(self, src, dst):
        subprocess.run(f'adb -s {self.serino} pull {src} {dst}', shell=True)

    def remove(self, dst):
        subprocess.run(f'adb -s {self.serino} shell rm {dst}', shell=True)

    def run_cmd(self, cmd):
        result = subprocess.check_output(f'adb -s {self.serino} shell {cmd}', shell=True).decode('utf-8')
        print(result)
        return result

class TfliteCnnTester:
    def __init__(self, adb: ADB, cnn_zoo_dict: dict, model_zoo_dir: str):
        self.adb = adb
        self.cnn_zoo_dict = cnn_zoo_dict
        self.tf_model_dir = os.path.join(model_zoo_dir, 'tf_model', 'quant_cnn_test')
        self.tflite_model_dir = os.path.join(model_zoo_dir, 'tflite_model', 'quant_cnn_test')

    def _get_tf_path(self, model_name) -> str:
        return os.path.join(self.tf_model_dir, model_name+'.tf')
    
    def _get_fp32_tflite_path(self, model_name) -> str:
        return os.path.join(self.tflite_model_dir, 'fp32', model_name+'.tflite')

    def _get_int8_tflite_path(self, model_name) -> str:
        return os.path.join(self.tflite_model_dir, 'int8', model_name+'_quant_int8.tflite')

    def _export_tf(self, ):
        print('===== Exporting TF Saved Model =====')
        for model_name, generator_func in self.cnn_zoo_dict.items():
            model = generator_func() 
            model.save(self._get_tf_path(model_name))

    def _convert(self, ):
        print('===== Converting TFLite =====')
        tf2tflite_dir(self.tf_model_dir, os.path.join(self.tflite_model_dir, 'fp32'))

    def _quantize(self, ):
        print('===== Quantizing =====')
        tf2tflite_dir(self.tf_model_dir, os.path.join(self.tflite_model_dir, 'int8'))

    def _fetch_latency(self, text: str, target='cpu_fp32'):
        if target in ['cpu_fp32', 'cpu_int8']:
            print(f'BEGIN {target}------------------')
            print(text)
            print('END------------------')

            match = re.findall(r'avg=\d+\.\d+|avg=\d+', text)[-1]
            return float(match[len('avg='): ]) / 1000
        else:
            match = re.findall(r'Total time - \d+\.\d+ms|Total time - \d+ms', text)[-1]
            return float(match[len('Total time - '): -len('ms')])

    def _benchmark_single(self, model_path, target='cpu_fp32'):
        assert target in ['cpu_fp32', 'cpu_int8', 'gpu_fp32', 'gpu_fp16']
        file_name = os.path.basename(model_path)
        dst_path = f'/sdcard/{file_name}'   
        avg_ms = 0.0
        self.adb.push(model_path, f'/sdcard/{file_name}')

        try:
            if target in ['cpu_fp32', 'cpu_int8']:
                output_text = self.adb.run_cmd(
                    f'taskset 70 /data/local/tmp/benchmark_model_plus_flex_r27 --graph={dst_path} --num_runs=30 --warmup_runs=10 --use_xnnpack=false --num_threads=1')
            else:
                output_text = self.adb.run_cmd(f'/data/local/tmp/performance_profiling_plus_f32 {dst_path} {"F32" if target == "gpu_fp32" else "F16"}')
        except:
            pass
        
        self.adb.remove(dst_path)
        avg_ms = self._fetch_latency(output_text, target)
        return avg_ms

    def _benchmark(self, ):
        print('===== Benchmarking =====')
        name_list = list(self.cnn_zoo_dict.keys())
        result_dict = {}
        for model_name in name_list:
            result_dict[model_name] = {}

        for target in ['cpu_fp32', 'gpu_fp32', 'gpu_fp16']:
            for model_name in name_list:
                tflite_path = self._get_fp32_tflite_path(model_name)
                avg_ms = self._benchmark_single(tflite_path, target)
                result_dict[model_name][target] = round(avg_ms, 2 if target == 'cpu' else 5)
        for model_name in name_list:
            tflite_path = self._get_int8_tflite_path(model_name)
            avg_ms = self._benchmark_single(tflite_path, 'cpu_int8')
            result_dict[model_name]['cpu_int8'] = round(avg_ms, 2)

        print('===============================')
        print('          SUMMARY')
        print('===============================')
        print(*name_list)
        for target in ['cpu_fp32', 'cpu_int8', 'gpu_fp32', 'gpu_fp16']:
            print(target, *[result_dict[k][target] for k in name_list])

    def run(self, ):
        self._benchmark()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_zoo_dir', default='models', help='root dir to save tf and tflite models')
    parser.add_argument('--serial_number', default='98281FFAZ009SV', help='phone serial number')
    args = parser.parse_args()

    adb = ADB(args.serial_number)
    tflite_cnn_tester = TfliteCnnTester(adb, cnn_zoo_dict, args.model_zoo_dir)
    tflite_cnn_tester.run()

if __name__ == '__main__':
    main()