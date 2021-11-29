import os
from re import X
import sys
import argparse

import tensorflow as tf
import tensorflow.keras.layers as layers

sys.path.append(os.path.join(os.path.dirname(sys.argv[0]), '..'))
import utils


def main():
    def make_model(input_shape, op, batch_size=1):
        input = tf.keras.Input(shape=input_shape, batch_size=batch_size)
        output = op(input)
        return tf.keras.Model(input, output)

    def quant_model(input_path, output_path: str, input_shape):
        assert output_path.endswith('.tflite')
        utils.tf2tflite(input_path, output_path)
        # tf2tflite(input_path, output_path.replace('.tflite', '_quant_float16.tflite').replace('/fp32/', '/fp16/'), quantization='float16')
        utils.tf2tflite(input_path, output_path.replace('.tflite', '_quant_int8.tflite').replace('/fp32/', '/int8/'), quantization='int8', input_shape=input_shape)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_zoo_dir', required=True, type=str, help='output tf model dir')
    args = parser.parse_args()

    tf_dir = os.path.join(args.model_zoo_dir, 'tf_model', 'quant_op_test')
    tflite_dir = os.path.join(args.model_zoo_dir, 'tflite_model', 'quant_op_test')

    # save dense models
    
    for x in range(160, 225):
        model = make_model([197, 192], layers.Dense(x))
        model.save(os.path.join(tf_dir, 'dense_out', f'dense197_192_{x}.tf'))
        quant_model(
            os.path.join(tf_dir, 'dense_out', f'dense197_192_{x}.tf'),
            os.path.join(tflite_dir, 'dense_out', 'fp32',  f'dense197_192_{x}.tflite'),
            [1, 197, 192]
        )
 
        model = make_model([197, x], layers.Dense(192))
        model.save(os.path.join(tf_dir, 'dense_in', f'dense197_{x}_192.tf'))
        quant_model(
            os.path.join(tf_dir, 'dense_in', f'dense197_{x}_192.tf'),
            os.path.join(tflite_dir, 'dense_in', 'fp32',  f'dense197_{x}_192.tflite'),
            [1, 197, x]
        )
    
    for x in range(1, 65):
        model = make_model([197, 192], layers.Dense(x))
        model.save(os.path.join(tf_dir, 'dense_out2', f'dense197_192_{x:02}.tf'))
        quant_model(
            os.path.join(tf_dir, 'dense_out2', f'dense197_192_{x:02}.tf'),
            os.path.join(tflite_dir, 'dense_out2', 'fp32',  f'dense197_192_{x:02}.tflite'),
            [1, 197, 192]
        )
 
        model = make_model([197, x], layers.Dense(192))
        model.save(os.path.join(tf_dir, 'dense_in2', f'dense197_{x:02}_192.tf'))
        quant_model(
            os.path.join(tf_dir, 'dense_in2', f'dense197_{x:02}_192.tf'),
            os.path.join(tflite_dir, 'dense_in2', 'fp32',  f'dense197_{x:02}_192.tflite'),
            [1, 197, x]
        )

    for x in range(1, 198):
        model = make_model([x, 768], layers.Dense(3072))
        model.save(os.path.join(tf_dir, 'dense_n', f'dense{x:03}_768_3072.tf'))
        quant_model(
            os.path.join(tf_dir, 'dense_n', f'dense{x:03}_768_3072.tf'),
            os.path.join(tflite_dir, 'dense_n', 'fp32', f'dense{x:03}_768_3072.tflite'),
            [1, x, 768]
        )

    for n in [8, 16, 197]:
        for i in [192, 768, 3072]:
            for h in [32, 192, 768]:
                model = make_model([h], layers.Dense(i), batch_size=n)
                model.save(os.path.join(tf_dir, 'dense_nhi2d', f'dense{n}_{h}_{i}.tf'))
                quant_model(
                    os.path.join(tf_dir, 'dense_nhi2d', f'dense{n}_{h}_{i}.tf'),
                    os.path.join(tflite_dir, 'dense_nhi2d', 'fp32', f'dense{n}_{h}_{i}.tflite'),
                    [n, h]
                )
    
    for x in range(0, 13):
        model = make_model([197, 768], layers.Dense(1<<x))
        model.save(os.path.join(tf_dir, 'dense_out3', f'dense197_768_{(1<<x):04}.tf'))
        quant_model(
            os.path.join(tf_dir, 'dense_out3', f'dense197_768_{(1<<x):04}.tf'),
            os.path.join(tflite_dir, 'dense_out3', 'fp32',  f'dense197_768_{(1<<x):04}.tflite'),
            [1, 197, 768]
        )

        model = make_model([197, 1<<x], layers.Dense(768))
        model.save(os.path.join(tf_dir, 'dense_in3', f'dense197_{(1<<x):04}_768.tf'))
        quant_model(
            os.path.join(tf_dir, 'dense_in3', f'dense197_{(1<<x):04}_768.tf'),
            os.path.join(tflite_dir, 'dense_in3', 'fp32',  f'dense197_{(1<<x):04}_768.tflite'),
            [1, 197, 1<<x]
        )

        model = make_model([1<<x, 768], layers.Dense(768))
        model.save(os.path.join(tf_dir, 'dense_n3', f'dense{(1<<x):04}_768_768.tf'))
        quant_model(
            os.path.join(tflite_dir, 'dense_n3', f'dense{(1<<x):04}_768_768.tf'),
            os.path.join(tflite_dir, 'dense_n3', 'fp32', f'dense{(1<<x):04}_768_768.tflite'),
            [1, 1<<x, 768]
        )

    # save conv
    for x in [1, 3, 5, 7]:
        model = make_model([56, 56, 32], layers.Conv2D(32, x, padding='same'))
        model.save(os.path.join(tf_dir, 'conv_kernel', f'conv_k{x}_i32_o32_hw56.tf'))
        quant_model(
            os.path.join(tf_dir, 'conv_kernel', f'conv_k{x}_i32_o32_hw56.tf'),
            os.path.join(tflite_dir, 'conv_kernel', 'fp32', f'conv_k{x}_i32_o32_hw56.tflite'),
            [1, 56, 56, 32]
        )


    for x in [1, 3, 5, 7]:
        model = make_model([14, 14, 320], layers.Conv2D(320, x, padding='same'))
        model.save(os.path.join(tf_dir, 'conv_kernel2', f'conv_k{x}_i320_o320_hw14.tf'))
        quant_model(
            os.path.join(tf_dir, 'conv_kernel2', f'conv_k{x}_i320_o320_hw14.tf'),
            os.path.join(tflite_dir, 'conv_kernel2', 'fp32', f'conv_k{x}_i320_o320_hw14.tflite'),
            [1, 14, 14, 320]
        )
    
    for x in range(1, 128 + 1):
        model = make_model([56, 56, 32], layers.Conv2D(x, 3, padding='same'))
        model.save(os.path.join(tf_dir, 'conv_cout', f'conv_k3_i32_o{x:03}_hw56.tf'))
        quant_model(
            os.path.join(tf_dir, 'conv_cout', f'conv_k3_i32_o{x:03}_hw56.tf'),
            os.path.join(tflite_dir, 'conv_cout', 'fp32', f'conv_k3_i32_o{x:03}_hw56.tflite'),
            [1, 56, 56, 32]
        )

    for x in range(1, 128 + 1):
        model = make_model([56, 56, x], layers.Conv2D(32, 3, padding='same'))
        model.save(os.path.join(tf_dir, 'conv_cin', f'conv_k3_i{x:03}_o32_hw56.tf'))
        quant_model(
            os.path.join(tf_dir, 'conv_cin', f'conv_k3_i{x:03}_o32_hw56.tf'),
            os.path.join(tflite_dir, 'conv_cin', 'fp32', f'conv_k3_i{x:03}_o32_hw56.tflite'),
            [1, 56, 56, x]
        )

    # save dwconv
    for x in [1, 3, 5, 7]:
        model = make_model([56, 56, 32], layers.DepthwiseConv2D(kernel_size=x, padding='same', depth_multiplier=1))
        model.save(os.path.join(tf_dir, 'dwconv_kernel', f'dwconv_k{x}_i32_o32_hw56.tf'))
        quant_model(
            os.path.join(tf_dir, 'dwconv_kernel', f'dwconv_k{x}_i32_o32_hw56.tf'),
            os.path.join(tflite_dir, 'dwconv_kernel', 'fp32', f'dwconv_k{x}_i32_o32_hw56.tflite'),
            [1, 56, 56, 32]
        )
    for x in [1, 3, 5, 7]:
        model = make_model([14, 14, 320], layers.DepthwiseConv2D(kernel_size=x, padding='same', depth_multiplier=1))
        model.save(os.path.join(tf_dir, 'dwconv_kernel2', f'dwconv_k{x}_i320_o320_hw14.tf'))
        quant_model(
            os.path.join(tf_dir, 'dwconv_kernel2', f'dwconv_k{x}_i320_o320_hw14.tf'),
            os.path.join(tflite_dir, 'dwconv_kernel2', 'fp32', f'dwconv_k{x}_i320_o320_hw14.tflite'),
            [1, 14, 14, 320]
        )

    for x in range(16, 49):
        model = make_model([56, 56, x], layers.DepthwiseConv2D(kernel_size=3, padding='same', depth_multiplier=1))
        model.save(os.path.join(tf_dir, 'dwconv_cio', f'dwconv_k3_io{x}_hw56.tf'))
        quant_model(
            os.path.join(tf_dir, 'dwconv_cio', f'dwconv_k3_io{x}_hw56.tf'),
            os.path.join(tflite_dir, 'dwconv_cio', 'fp32', f'dwconv_k3_io{x}_hw56.tflite'),
            [1, 56, 56, x]
        )


    # save relu
    for x in range(16, 49):
        model = make_model([56, 56, x], layers.ReLU())
        model.save(os.path.join(tf_dir, 'relu', f'relu_hw56_c{x}.tf'))
        quant_model(
            os.path.join(tf_dir, 'relu', f'relu_hw56_c{x}.tf'),
            os.path.join(tflite_dir, 'relu', 'fp32', f'relu_hw56_c{x}.tflite'),
            [1, 56, 56, x]
        )


if __name__ == '__main__':
    main()