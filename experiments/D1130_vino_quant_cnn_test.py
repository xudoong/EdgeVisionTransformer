import os
import sys
import argparse
import subprocess

sys.path.insert(0, f'{os.path.dirname(sys.argv[0])}/..')
from benchmark.openvino.vino_cli import openvino_benchmark

MODEL_LIST = [
    'mobilenet-v1-1.0-224-tf',
    'mobilenet-v2-1.0-224',
    'shufflenet-v2-x1.0',
    'inception-resnet-v2-tf',
    'efficientnet-b0',
    'resnet-34-pytorch',
    'resnet-50-tf', 
]


class ModelExporter:
    def __init__(self, model_list, download_model_dir, ir_model_dir):
        self.model_list = model_list
        self.download_model_dir = download_model_dir
        self.ir_model_dir = ir_model_dir
        pass

    def download(self):
        subprocess.run(
            f'python $VINO_DOWNLOADER --name={",".join(self.model_list)} --precisions=FP32 --output_dir={self.download_model_dir}', shell=True)

    def convert(self):
        subprocess.run(
            f'python $VINO_CONVERTER --name={",".join(self.model_list)} --precisions=FP32 --download_dir={self.download_model_dir} --output_dir={self.ir_model_dir}', shell=True)

    def quantize(self):
        subprocess.run(
            f'python $VINO_QUANTER --name={",".join(self.model_list)} --model_dir={self.ir_model_dir} --dataset_dir=datasets/imagenet2012 --output_dir={self.ir_model_dir} --precisions=FP32-INT8', shell=True)

    def benchmark(self):
        print('========== Benchmarking model performance on CPU with 1 thread')
        latency_list_fp32 = []
        latency_list_int8 = []

        for name in self.model_list:
            print(f'===== Testing {name} =====')
            model_path_fp32 = os.path.join(self.ir_model_dir, 'public', name, 'FP32', f'{name}.xml')
            model_path_int8 = os.path.join(self.ir_model_dir, 'public', name, 'FP32-INT8', f'{name}.xml')

            latency_fp32 = openvino_benchmark('$VINO_BENCHMARK_APP', model_path_fp32, niter=30, num_threads=1, batch_size=1)
            latency_int8 = openvino_benchmark('$VINO_BENCHMARK_APP', model_path_int8, niter=30, num_threads=1, batch_size=1)

            latency_list_fp32.append(latency_fp32)
            latency_list_int8.append(latency_int8)

        print('== SUMMARY ==')
        print(self.model_list)
        print('FP32 Latency:')
        print([round(v, 2) for v in latency_list_fp32])
        print('INT8 Latency:')
        print([round(v, 2) for v in latency_list_int8])


if __name__ == '__main__':
    model_exporter = ModelExporter(
        MODEL_LIST, 'models/vino_model/download', 'models/vino_model/ir')
    model_exporter.download()
    model_exporter.convert()
    model_exporter.quantize()
    model_exporter.benchmark()