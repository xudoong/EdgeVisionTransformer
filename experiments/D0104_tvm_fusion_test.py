import argparse
import os
from pathlib import Path
import re
import subprocess


OUTPUT_CSV_PATH = 'D0105_tvm_fusion_test.csv'


class TvmFusionTester:
    def __init__(self, tflite_dir, tvm_dir, tracker_host, tracker_port, rpc_key, cross_compiler_path):
        self.tflite_dir = tflite_dir
        self.tvm_dir=tvm_dir
        self.tracker_host = tracker_host
        self.tracker_port = tracker_port
        self.rpc_key = rpc_key
        self.cross_compiler_path = cross_compiler_path

    def _compile_single(self, input_path, output_path):
        Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
        cmd = f'tvmc compile {input_path} -o {output_path} --target "llvm -model=snapdragon855 -mtriple=arm64-linux-android -mattr=+neon" --cross-compiler {self.cross_compiler_path}'
        result = subprocess.check_output(cmd, shell=True).decode('utf-8')
        print(result)

    def _benchmark_single(self, model_path):
        print(os.path.basename(model_path))
        result = subprocess.run(
            f'tvmc run {model_path} --rpc-key {self.rpc_key} --rpc-tracker {self.tracker_host}:{self.tracker_port} --print-time --device=cpu --repeat 100',
            shell=True, capture_output=True).stdout.decode('utf-8')
        print(result)
        numbers = re.findall(r'\d*\.?\d+', result)
        return float(numbers[1]) # median time

    def _compile(self):
        print('==== Compiling ====')
        for root, dirs, files in os.walk(self.tflite_dir):
            for file in sorted(files):
                if file.endswith('.tflite'):
                    input_path = os.path.join(root, file)
                    output_path = os.path.join(self.tvm_dir, input_path.replace(self.tflite_dir + '/', '').replace('.tflite', '.tar'))
                    self._compile_single(input_path, output_path)

    def _tune(self):
        print('==== Tuning ====')
        for root, dirs, files in os.walk(self.tflite_dir):
            for file in sorted(files):
                if file.endswith('.tflite'):
                    input_path = os.path.join(root, file)
                    output_path = os.path.join(self.tvm_dir, input_path.replace(self.tflite_dir + '/', '').replace('.tflite', '_autotuner_records.json'))
                    self._tune_single(input_path, output_path)

    def _benchmark(self):
        with open(OUTPUT_CSV_PATH, 'a') as f:
            f.write('model_name,avg_ms\n')

        print('==== benchmarking ====')
        for root, dirs, files in os.walk(self.tvm_dir):
            for file in sorted(files):
                if file.endswith('.tar'):
                    model_path = os.path.join(root, file)
                    avg_ms = self._benchmark_single(model_path)
                    with open(OUTPUT_CSV_PATH, 'a') as f:
                        f.write(f'{os.path.basename(model_path)},{avg_ms:.2f}\n')

    def run(self):
        self._compile()
        self._benchmark()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tflite_dir', default='models/tflite_model/fusion', type=str)
    parser.add_argument('--tvm_dir', default='models/tvm_model/fusion_test', type=str)
    parser.add_argument('--tracker_host', default='127.0.0.1', type=str)
    parser.add_argument('--tracker_port', default=9090, type=int)
    parser.add_argument('--rpc_key', default='android', type=str)
    parser.add_argument('--cross_compiler_path', default=os.environ['TVM_NDK_CC'], type=str)
    args = parser.parse_args()

    tester = TvmFusionTester(args.tflite_dir, args.tvm_dir, args.tracker_host, args.tracker_port, args.rpc_key, args.cross_compiler_path)
    tester.run()

if __name__ == '__main__':
    main()
