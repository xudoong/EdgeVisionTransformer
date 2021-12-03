import argparse
import os
import subprocess
import tensorflow as tf
import tensorflow.keras.layers as layers
import re
import sys
import numpy as np

sys.path.insert(0, f'{os.path.dirname(sys.argv[0])}/..')
from utils import tf2tflite

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
        result = subprocess.run(f'adb -s {self.serino} shell {cmd}', shell=True, capture_output=True).stdout.decode('utf-8')
        print(result)
        return result

class OpTester:
    def __init__(self, model_zoo_dir: str, serino: str, print_precision):
        self.tf_model_dir = os.path.join(model_zoo_dir, 'tf_model', 'gpu_op_test')
        self.tflite_model_dir = os.path.join(model_zoo_dir, 'tflite_model', 'gpu_op_test')
        self.adb = ADB(serino)
        self.print_precision = print_precision
        self.model_zoo = {
            # dense 
            'dense2d_out': {
                'name': 'dense2d197_192_{:03}',
                'range': range(1, 129),
                'param': [197, 192, None]
            },
            'dense2d_in': {
                'name': 'dense2d197_{:03}_192',
                'range': range(1, 129),
                'param': [197, None, 192]
            },
            'dense2d_n': {
                'name': 'dense2d{:03}_192_192',
                'range': range(1, 129),
                'param': [None, 192, 192]
            },
            'dense2d_out2': {
                'name': 'dense2d197_768_{:04}',
                'range': [1<<x for x in range(13)],
                'param': [197, 768, None]
            },
            'dense2d_in2': {
                'name': 'dense2d197_{:04}_768',
                'range': [1<<x for x in range(13)],
                'param': [197, None, 768]
            },
            'dense2d_n2': {
                'name': 'dense2d{:04}_768_768',
                'range': [1<<x for x in range(13)],
                'param': [None, 768, 768]
            },
            # conv
            'conv_kernel': {
                'name': 'conv_k{:01}_i32_o32_hw56',
                'range': [1, 3, 5, 7],
                'param': [32, 32, None, 56], # [cin, cout, kernel_size, hw]
            },
            'conv_kernel2': {
                'name': 'conv_k{:01}_i320_o320_hw14',
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
            }
        }

    def _add_input(self, input_shape, op, batch_size=1):
        input = tf.keras.Input(shape=input_shape, batch_size=batch_size)
        output = op(input)
        return tf.keras.Model(input, output)

    def _export_single(self, test_case, v, i):
        name = v['name'].format(i)
        param = [x or i for x in v['param']]

        if re.match(r'dense2d', test_case):
            model = self._add_input([param[1]], layers.Dense(param[2]), param[0])
        elif re.match(r'conv', test_case):
            model = self._add_input([param[3], param[3], param[0]], layers.Conv2D(filters=param[1], kernel_size=param[2], padding='same'))
        elif re.match(r'dwconv', test_case):
            model = self._add_input([param[3], param[3], param[0]], layers.DepthwiseConv2D(depth_multiplier=param[1], kernel_size=param[2], padding='same'))
        else:
            raise ValueError(f'Only Dense, Conv2D and DepthWiseConv2d are supported.')

        output_path_tf = os.path.join(self.tf_model_dir, test_case, f'{name}.tf')
        output_path_tflite = os.path.join(self.tflite_model_dir, test_case, f'{name}.tflite')
        model.save(output_path_tf)
        tf2tflite(output_path_tf, output_path_tflite)

    def _export_tflite(self, ):
        print('===== Exporting TFLite =====')
        for test_case, v in self.model_zoo.items():
            for i in v['range']:
                self._export_single(test_case, v, i)

    def _fetch_latency(self, text):
        match_list = re.findall(r'Total time - \d+\.\d+ms', text)
        latency_list = [float(x[len('Total time - '): -len('ms')]) for x in match_list]
        latency_list = sorted(latency_list)
        return np.average(latency_list[: max(1, len(latency_list) // 3)])

    def _benchmark_single(self, model_path):
        file_name = os.path.basename(model_path)
        dst_path = f'/sdcard/{file_name}'   
        fp32_ms = 0.0
        fp16_ms = 0.0
        try:
            self.adb.push(model_path, f'/sdcard/{file_name}')
            fp32_output = self.adb.run_cmd(f'/data/local/tmp/performance_profiling_plus_f32 {dst_path} F32')
            fp16_output = self.adb.run_cmd(f'/data/local/tmp/performance_profiling_plus_f32 {dst_path} F16')
            self.adb.remove(dst_path)
            fp32_ms = self._fetch_latency(fp32_output)
            fp16_ms = self._fetch_latency(fp16_output)
        except:
            pass
        print(fp32_ms, fp16_ms)
        return fp32_ms, fp16_ms

    def _benchmark(self, ):
        print('====== Benchmarking =====')
        result_summary = {}
        for test_case, v in self.model_zoo.items():
            print(f'==    Testing {test_case}    ==')
            result_summary[test_case] = {}
            latency_list_fp32 = []
            latency_list_fp16 = []
            for i in v['range']:
                model_name = v['name'].format(i)
                model_path = os.path.join(self.tflite_model_dir, test_case, f'{model_name}.tflite')
  
                latency_fp32, latency_fp16 = self._benchmark_single(model_path)

                latency_list_fp32.append(round(latency_fp32, self.print_precision))
                latency_list_fp16.append(round(latency_fp16, self.print_precision))
            print(f'--- {test_case} Summary ---')
            print('FP32 ms')
            print(latency_list_fp32)
            print('FP16 ms')
            print(latency_list_fp16)
            result_summary[test_case]['fp32'] = latency_list_fp32
            result_summary[test_case]['fp16'] = latency_list_fp16

        print('== SUMMARY ==')
        for test_case, v in result_summary.items():
            print(f'--- {test_case} ---')
            print('FP32 ms')
            print(v['fp32'])
            print('FP16 ms')
            print(v['fp16'])

    def run(self, ):
        # self._export_tflite()
        self._benchmark()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_zoo_dir', default='models', type=str, help='tf and tflite model dir')
    parser.add_argument('--serino', default='98281FFAZ009SV', type=str, help='phone serial number to test')
    parser.add_argument('--print_precision', default=4, type=int, help='precision to print latency')
    args = parser.parse_args()

    tester = OpTester(args.model_zoo_dir, args.serino, args.print_precision)
    tester.run()


if __name__ == '__main__':
    main()