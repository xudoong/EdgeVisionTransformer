import torch
import os
import sys
import argparse
import numpy as np
from torch2trt import torch2trt


def mkdirs_ifnot_exist(path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))


class TrtOpTester:
    def __init__(self, model_zoo_dir):
        self.output_dir = os.path.join(model_zoo_dir, 'trt_model', 'quant_op_test')

    def _add_input_shape_to_state_dict(self, model, input_shape):
        return {'input_shape': input_shape, 'model': model.state_dict()}

    def _save_model(self, model, fp32_output_path, input_shape):
        model = model.eval().cuda()
        dummy_input = torch.randn(input_shape).cuda()
        int8_output_path = fp32_output_path.replace('/fp32/', '/int8/').replace('.pth', '_quant_int8.pth')
        mkdirs_ifnot_exist(fp32_output_path)
        mkdirs_ifnot_exist(int8_output_path)

        # save fp32 model
        model_fp32_trt = torch2trt(model, [dummy_input])
        torch.save(self._add_input_shape_to_state_dict(model_fp32_trt, input_shape), fp32_output_path)
        # save int8 model
        model_int8_trt = torch2trt(model, [dummy_input], int8_mode=True)
        torch.save(self._add_input_shape_to_state_dict(model_int8_trt, input_shape), int8_output_path)

        print(f'Save {fp32_output_path} done.')

    def _save_dense(self):
        # === varying output size ===
        for x in range(192, 192 * 6 + 1, 192):
            model = torch.nn.Linear(in_features=192, out_features=x)
            output_path = os.path.join(self.output_dir, 'dense_out', 'fp32', f'dense197_192_{x}.pth')
            self._save_model(model, output_path, [1, 197, 192])

        # === varying input size ===
        for x in range(160, 224 + 1):
            model = torch.nn.Linear(in_features=x, out_features=192)
            output_path = os.path.join(self.output_dir, 'dense_in', 'fp32', f'dense197_{x}_192.pth')
            self._save_model(model, output_path, [1, 197, x])

        # === varying seq_len ===
        for x in range(1, 64):
            model = torch.nn.Linear(in_features=768, out_features=3072)
            output_path = os.path.join(self.output_dir, 'dense_n', 'fp32', f'dense{x:02}_768_3072.pth')
            self._save_model(model, output_path, [1, x, 768])


        # === varying output size (2nd range) ===
        for x in range(1, 64 + 1):
            model = torch.nn.Linear(in_features=192, out_features=x)
            output_path = os.path.join(self.output_dir, 'dense_out2', 'fp32', f'dense197_192_{x:02}.pth')
            self._save_model(model, output_path, [1, 197, 192])

        # === varying input size (2nd range) ===
        for x in range(1, 64 + 1):
            model = torch.nn.Linear(in_features=x, out_features=192)
            output_path = os.path.join(self.output_dir, 'dense_in2', 'fp32', f'dense197_{x:02}_192.pth')
            self._save_model(model, output_path, [1, 197, x])

        # === varying all 3 ===
        for n in [8, 16, 197]:
            for i in [192, 768, 3072]:
                for h in [32, 192, 768]:
                    model = torch.nn.Linear(in_features=h, out_features=i)
                    output_path = os.path.join(self.output_dir, 'dense_group', 'fp32', f'dense{n:03}_{h:03}_{i:04}.pth')
                    self._save_model(model, output_path, [1, n, h])

    def _save_conv(self):
        # === varying kernel size ===
        for x in [1, 3, 5, 7]:
            model = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=x, padding=(x-1)//2)
            output_path = os.path.join(self.output_dir, 'conv_kernel', 'fp32', f'conv_k{x}_i32_o32_hw56.pth')
            self._save_model(model, output_path, [1, 32, 56, 56])

        # === varying kernel size (2nd config) ===
        for x in [1, 3, 5, 7]:
            model = torch.nn.Conv2d(in_channels=320, out_channels=320, kernel_size=x, padding=(x-1)//2)
            output_path = os.path.join(self.output_dir, 'conv_kernel2', 'fp32', f'conv_k{x}_i320_o320_hw14.pth')
            self._save_model(model, output_path, [1, 320, 14, 14])

        # === varying output channel size ===
        for x in range(1, 72 + 1):
            model = torch.nn.Conv2d(in_channels=32, out_channels=x, kernel_size=3, padding=1)
            output_path = os.path.join(self.output_dir, 'conv_cout', 'fp32', f'conv_k3_i32_o{x:03}_hw56.pth')
            self._save_model(model, output_path, [1, 32, 56, 56])

    def _save_dwconv(self):
        # === varying kernel size ===
        for x in [1, 3, 5, 7]:
            model = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=x, padding=(x-1)//2, groups=32)
            output_path = os.path.join(self.output_dir, 'dwconv_kernel', 'fp32', f'dwconv_k{x}_io32_hw56.pth')
            self._save_model(model, output_path, [1, 32, 56, 56])

        # === varying kernel size (2nd config) ===
        for x in [1, 3, 5, 7]:
            model = torch.nn.Conv2d(in_channels=320, out_channels=320, kernel_size=x, padding=(x-1)//2, groups=320)
            output_path = os.path.join(self.output_dir, 'dwconv_kernel2', 'fp32', f'dwconv_k{x}_io320_hw14.pth')
            self._save_model(model, output_path, [1, 320, 14, 14])

        # === varying output channel size ===
        for x in range(1, 72 + 1):
            model = torch.nn.Conv2d(in_channels=x, out_channels=x, kernel_size=3, padding=1, groups=x)
            output_path = os.path.join(self.output_dir, 'dwconv_cout', 'fp32', f'dwconv_k3_io{x:03}_hw56.pth')
            self._save_model(model, output_path, [1, x, 56, 56])

    def save_models(self):
        self._save_dense()
        self._save_conv()
        self._save_dwconv()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_zoo_dir', type=str, required=True, help='model zoo dir')
    args = parser.parse_args()

    tester = TrtOpTester(args.model_zoo_dir)
    tester.save_models()

if __name__ == '__main__':
    main()
