import torch
from onnxruntime.quantization import quantize_dynamic, QuantType
import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(sys.argv[0]), '..'))
import utils


def mkdirs_ifnot_exist(path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))


class OnnxOpTester:
    def __init__(self, model_zoo_dir):
        self.output_dir = os.path.join(model_zoo_dir, 'onnx_model', 'quant_op_test')

    def _save_model(self, model, fp32_output_path, input_shape):
        utils.export_onnx_fix_batch(model, fp32_output_path, input_shape)
        
        quantize_dynamic_output_path = fp32_output_path.replace('/fp32/', '/dynamic/').replace('.onnx', '_quant_dynamic.onnx')
        mkdirs_ifnot_exist(quantize_dynamic_output_path)
        quantize_dynamic(fp32_output_path, quantize_dynamic_output_path, activation_type=QuantType.QUInt8)

    def _save_dense(self):
        # === varying output size ===
        for x in range(160, 225):
            model = torch.nn.Linear(in_features=192, out_features=x)
            output_path = os.path.join(self.output_dir, 'dense_out', 'fp32', f'dense197_192_{x}.onnx')
            self._save_model(model, output_path, [197, 192])

    def save_models(self):
        self._save_dense()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_zoo_dir', type=str, required=True, help='model zoo dir')
    args = parser.parse_args()

    tester = OnnxOpTester(args.model_zoo_dir)
    tester.save_models()

if __name__ == '__main__':
    main()