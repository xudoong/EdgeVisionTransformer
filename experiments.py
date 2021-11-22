import tensorflow as tf
import tensorflow.keras.layers as layers
import os
import sys
import argparse

from modeling.layers.activation import gelu
from modeling.models.vit import ViT_Pruned
from utils import add_keras_input_layer



class ModelGenerator():
    def __init__(self, output_tf_dir, output_tflite_dir):
        self.output_tf_dir = output_tf_dir
        self.output_tflite_dir = output_tflite_dir
        self.model_dict = {}

    def save_model(self):
        for k, v in self.model_dict.items():
            dest = os.path.join(self.output_tf_dir, f'{k}.tf')
            v.save(dest)
            print(f'Save model to {dest}.')

    def convert_to_tflite(self):
        from utils import tf2tflite
        for k, v in self.model_dict.items():
            src = os.path.join(self.output_tf_dir, f'{k}.tf')
            dst = os.path.join(self.output_tflite_dir, f'{k}.tflite')
            tf2tflite(src, dst)

class FusionTestTransformer(ModelGenerator):
    def __init__(self, output_tf_dir, output_tflite_dir, l=197, h=768, i=3072) -> None:
        super().__init__(output_tf_dir, output_tflite_dir)

        self.l = l
        self.h = h
        self.i = i
        self.dense_lhi = add_keras_input_layer(layers.Dense(self.i), [self.l, self.h])
        self.dense_lhi_gelu = add_keras_input_layer(layers.Dense(self.i, activation=gelu), [self.l, self.h])
        self.gelu_li = self._get_gelu_li()

        self.dense_lhh = add_keras_input_layer(layers.Dense(self.h), [self.l, self.h])
        self.add_lh = self._get_add_lh()
        self.dense_lhh_add = self._get_dense_lhh_add()

        self.add_lh_layernorm = self._get_add_lh_layernorm()
        self.layernorm_lh = add_keras_input_layer(layers.LayerNormalization(epsilon=1e-5), [self.l, self.h])

        self.model_dict = {
            f'dense{self.l}_{self.h}_{self.i}' : self.dense_lhi,
            f'dense{self.l}_{self.h}_{self.i}_gelu' : self.dense_lhi_gelu,
            f'gelu{self.l}_{self.i}' : self.gelu_li,
            f'dense{self.l}_{self.h}_{self.h}' : self.dense_lhh,
            f'add{self.l}_{self.h}' : self.add_lh, 
            f'dense{self.l}_{self.h}_{self.h}_add' : self.dense_lhh_add,
            f'add{self.l}_{self.h}_layernorm' : self.add_lh_layernorm,
            f'layernorm{self.l}_{self.h}' : self.layernorm_lh
        }

    def _get_gelu_li(self):
        input = layers.Input([self.l, self.i])
        output = gelu(input)
        model = tf.keras.Model(input, output)
        return model

    def _get_dense_lhh_add(self):
        input = layers.Input([self.l, self.h])
        x = layers.Dense(self.h)(input)
        x = input + x
        model = tf.keras.Model(input, x)
        return model

    def _get_add_lh(self):
        input1 = layers.Input([self.l, self.h])
        input2 = layers.Input([self.l, self.h])
        output = input1 + input2
        model = tf.keras.Model([input1, input2], output)
        return model

    def _get_add_lh_layernorm(self):
        input1 = layers.Input([self.l, self.h])
        input2 = layers.Input([self.l, self.h])
        x = input1 + input2
        x = layers.LayerNormalization(epsilon=1e-5)(x)
        model = tf.keras.Model([input1, input2], x)
        return model
    

class FusionTestConv(ModelGenerator):
    def __init__(self, output_tf_dir, output_tflite_dir):
        super().__init__(output_tf_dir, output_tflite_dir)
        
        self.model_dict = {
            'conv_bn_relu': self._get_conv_bn_relu(),
            'conv_bn': self._get_conv_bn(),
            'conv_relu': self._get_conv_relu(),
            'conv': self._get_conv(),
            'bn': self._get_bn(),
            'relu': self._get_relu()
        }
    
    def _get_conv_bn_relu(self, ):
        model = tf.keras.Sequential([
            layers.Input([224, 224, 3]),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(2, 2)),
            tf.keras.layers.BatchNormalization(),
            layers.ReLU(6)
        ])
        return model

    def _get_conv_bn(self, ):
        model = tf.keras.Sequential([
            layers.Input([224, 224, 3]),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(2, 2)),
            tf.keras.layers.BatchNormalization(),
        ])
        return model

    def _get_conv_relu(self, ):
        model = tf.keras.Sequential([
            layers.Input([224, 224, 3]),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(2, 2)),
            layers.ReLU(6)
        ])
        return model

    def _get_conv(self, ):
        model = tf.keras.Sequential([
            layers.Input([224, 224, 3]),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(2, 2)),
        ])
        return model

    def _get_bn(self, ):
        model = tf.keras.Sequential([
            layers.Input([112, 112, 32]),
            tf.keras.layers.BatchNormalization(),
        ])
        return model

    def _get_relu(self, ):
        model = tf.keras.Sequential([
            layers.Input([224, 224, 3]),
            layers.ReLU(6)
        ])
        return model


class PruneBenchmark(ModelGenerator):
    def __init__(self, output_tf_dir, output_tflite_dir):
        super().__init__(output_tf_dir, output_tflite_dir)
        self.num_heads_dict = dict(tiny=3, small=6, base=12)
        # TODO: This impl is memory conusing. It is easy to run into OOM error.
        self._add_ffn_only_models()
        self._add_head_only_models()
        self._add_head_ffn_models()


    def _get_deit_config_from_num_heads(self, num_heads):
            hidden_size = num_heads * 64
            intermediate_size = hidden_size * 4
            return hidden_size, intermediate_size

    def _add_to_model_dict(self, type, prune_encoding, num_heads, intermediate_size, hidden_size):
            model = ViT_Pruned(dim=hidden_size, depth=12, heads=num_heads, mlp_dim=intermediate_size,
                                head_size=64, prune_encoding=prune_encoding)
            model = add_keras_input_layer(model, [3, 224, 224], 1)
            self.model_dict[f'deit_{type}_b1_{prune_encoding}'] = model

    def _add_ffn_only_models(self):
        for type in ['tiny', 'small', 'base']:
            num_heads = self.num_heads_dict[type]
            hidden_size, intermediate_size = self._get_deit_config_from_num_heads(num_heads)

            for thres in range(10, 100, 10):
                thres /= 100
                prune_encoding = f'all_head{num_heads}_ffn{thres}'
                self._add_to_model_dict(type=type, prune_encoding=prune_encoding, num_heads=num_heads,
                                        intermediate_size=intermediate_size, hidden_size=hidden_size)

    def _add_head_only_models(self):
        for type in ['tiny', 'small', 'base']:
            num_heads = self.num_heads_dict[type]
            hidden_size, intermediate_size = self._get_deit_config_from_num_heads(num_heads)

            for thres in range(1, num_heads):
                prune_encoding = f'all_head{thres}_ffn1.0'
                self._add_to_model_dict(type=type, prune_encoding=prune_encoding, num_heads=num_heads,
                                        intermediate_size=intermediate_size, hidden_size=hidden_size)

    def _add_head_ffn_models(self):
        prune_encodings = {
            'tiny': [f'all_head2_ffn{x}' for x in [0.7, 0.8, 0.9]],
            'small': [f'all_head{i}_ffn{j}' for i in [4, 5] for j in [0.6, 0.7, 0.8, 0.9]]
        }

        for type in prune_encodings.keys():
            num_heads = self.num_heads_dict[type]
            hidden_size, intermediate_size = self._get_deit_config_from_num_heads(num_heads)

            for prune_encoding in prune_encodings[type]:
                self._add_to_model_dict(type=type, prune_encoding=prune_encoding, num_heads=num_heads,
                                        intermediate_size=intermediate_size, hidden_size=hidden_size)


def fusion_test(parser: argparse.ArgumentParser):
    parser.add_argument('--output_tf_dir', required=True, type=str, help='output tf saved model dir')
    parser.add_argument('--output_tflite_dir', required=True, type=str, help='output tflite model dir')
    parser.add_argument('--type', choices=['tiny', 'base', 'conv'], help='this argument determines the op size')
    args = parser.parse_args()
    
    if args.type == 'base':
        l, h, i = 197, 768, 768 * 4
    elif args.type == 'tiny':
        l, h, i = 197, 192, 192 * 4

    if args.type == 'conv':
        fusion_test_class = FusionTestConv(args.output_tf_dir, args.output_tflite_dir)
    else:
        fusion_test_class = FusionTestTransformer(args.output_tf_dir, args.output_tflite_dir, l, h, i)
    fusion_test_class.save_model()
    fusion_test_class.convert_to_tflite()


def ncs2_test(parser: argparse.ArgumentParser):
    parser.add_argument('--output_tf_dir', required=True, type=str, help='output tf saved model dir')
    parser.add_argument('--output_tflite_dir', required=True, type=str, help='output tflite model dir')
    args = parser.parse_args()

    print('To generate transformer models w/o LayerNorm, models need to change temporarily at first.')

    from modeling.models.vit import get_deit_base, get_deit_small, get_deit_tiny
    from utils import add_keras_input_layer, tf2tflite

    model_dict = {
        'deit_base_patch16_224_noln': add_keras_input_layer(get_deit_base(), [3, 224, 224]),
        'deit_small_patch16_224_noln': add_keras_input_layer(get_deit_small(), [3, 224, 224]),
        'deit_tiny_patch16_224_noln': add_keras_input_layer(get_deit_tiny(), [3, 224, 224]),
    }

    for k, v in model_dict.items():
        dest = os.path.join(args.output_tf_dir, f'{k}.tf')
        v.save(dest)
        print(f'Save tf model to {dest}.')

    for k, v in model_dict.items():
        src = os.path.join(args.output_tf_dir, f'{k}.tf')
        dst = os.path.join(args.output_tflite_dir, f'{k}.tflite')
        tf2tflite(src, dst)


def prune_benchmark(parser: argparse.ArgumentParser):
    parser.add_argument('--output_tf_dir', required=True, type=str, help='output tf saved model dir')
    parser.add_argument('--output_tflite_dir', required=True, type=str, help='output tflite model dir')
    args = parser.parse_args()  

    prune_benchmarker = PruneBenchmark(args.output_tf_dir, args.output_tflite_dir)
    prune_benchmarker.save_model()


def quant_op_test(parser: argparse.ArgumentParser):
    from utils import tf2tflite
    def make_model(input_shape, op, batch_size=1):
        input = tf.keras.Input(shape=input_shape, batch_size=batch_size)
        output = op(input)
        return tf.keras.Model(input, output)

    def quant_model(input_path, output_path: str, input_shape):
        assert output_path.endswith('.tflite')
        tf2tflite(input_path, output_path)
        # tf2tflite(input_path, output_path.replace('.tflite', '_quant_float16.tflite').replace('/fp32/', '/fp16/'), quantization='float16')
        tf2tflite(input_path, output_path.replace('.tflite', '_quant_int8.tflite').replace('/fp32/', '/int8/'), quantization='int8', input_shape=input_shape)


    parser.add_argument('--model_zoo_dir', required=True, type=str, help='output tf model dir')
    args = parser.parse_args()

    tf_dir = os.path.join(args.model_zoo_dir, 'tf_model', 'quant_op_test')
    tflite_dir = os.path.join(args.model_zoo_dir, 'tflite_model', 'quant_op_test')

    # save dense models
    # for x in range(160, 225):
    #     model = make_model([197, 192], layers.Dense(x))
    #     model.save(os.path.join(tf_dir, 'dense_out', f'dense197_192_{x}.tf'))
    #     quant_model(
    #         os.path.join(tf_dir, 'dense_out', f'dense197_192_{x}.tf'),
    #         os.path.join(tflite_dir, 'dense_out', 'fp32',  f'dense197_192_{x}.tflite'),
    #         [1, 197, 192]
    #     )
 
    #     model = make_model([197, x], layers.Dense(192))
    #     model.save(os.path.join(tf_dir, 'dense_in', f'dense197_{x}_192.tf'))
    #     quant_model(
    #         os.path.join(tf_dir, 'dense_in', f'dense197_{x}_192.tf'),
    #         os.path.join(tflite_dir, 'dense_in', 'fp32',  f'dense197_{x}_192.tflite'),
    #         [1, 197, x]
    #     )
    
    # for x in range(1, 65):
    #     model = make_model([197, 192], layers.Dense(x))
    #     model.save(os.path.join(tf_dir, 'dense_out2', f'dense197_192_{x:02}.tf'))
    #     quant_model(
    #         os.path.join(tf_dir, 'dense_out2', f'dense197_192_{x:02}.tf'),
    #         os.path.join(tflite_dir, 'dense_out2', 'fp32',  f'dense197_192_{x:02}.tflite'),
    #         [1, 197, 192]
    #     )
 
    #     model = make_model([197, x], layers.Dense(192))
    #     model.save(os.path.join(tf_dir, 'dense_in2', f'dense197_{x:02}_192.tf'))
    #     quant_model(
    #         os.path.join(tf_dir, 'dense_in2', f'dense197_{x:02}_192.tf'),
    #         os.path.join(tflite_dir, 'dense_in2', 'fp32',  f'dense197_{x:02}_192.tflite'),
    #         [1, 197, x]
    #     )

    # for x in range(1, 198):
    #     model = make_model([x, 768], layers.Dense(3072))
    #     model.save(os.path.join(tf_dir, 'dense_n', f'dense{x:03}_768_3072.tf'))
    #     quant_model(
    #         os.path.join(tf_dir, 'dense_n', f'dense{x:03}_768_3072.tf'),
    #         os.path.join(tflite_dir, 'dense_n', 'fp32', f'dense{x:03}_768_3072.tflite'),
    #         [1, x, 768]
    #     )

    # for n in [8, 16, 197]:
    #     for i in [192, 768, 3072]:
    #         for h in [32, 192, 768]:
    #             model = make_model([h], layers.Dense(i), batch_size=n)
    #             model.save(os.path.join(tf_dir, 'dense_nhi2d', f'dense{n}_{h}_{i}.tf'))
    #             quant_model(
    #                 os.path.join(tf_dir, 'dense_nhi2d', f'dense{n}_{h}_{i}.tf'),
    #                 os.path.join(tflite_dir, 'dense_nhi2d', 'fp32', f'dense{n}_{h}_{i}.tflite'),
    #                 [n, h]
    #             )

                  

    # save conv
    # for x in [1, 3, 5, 7]:
    #     model = make_model([56, 56, 32], layers.Conv2D(32, x, padding='same'))
    #     model.save(os.path.join(tf_dir, 'conv_kernel', f'conv_k{x}_i32_o32_hw56.tf'))
    #     quant_model(
    #         os.path.join(tf_dir, 'conv_kernel', f'conv_k{x}_i32_o32_hw56.tf'),
    #         os.path.join(tflite_dir, 'conv_kernel', 'fp32', f'conv_k{x}_i32_o32_hw56.tflite'),
    #         [1, 56, 56, 32]
    #     )


    # for x in [1, 3, 5, 7]:
    #     model = make_model([14, 14, 320], layers.Conv2D(320, x, padding='same'))
    #     model.save(os.path.join(tf_dir, 'conv_kernel2', f'conv_k{x}_i320_o320_hw14.tf'))
    #     quant_model(
    #         os.path.join(tf_dir, 'conv_kernel2', f'conv_k{x}_i320_o320_hw14.tf'),
    #         os.path.join(tflite_dir, 'conv_kernel2', 'fp32', f'conv_k{x}_i320_o320_hw14.tflite'),
    #         [1, 14, 14, 320]
    #     )

    # for x in range(16, 49):
    #     model = make_model([56, 56, 32], layers.Conv2D(x, 3, padding='same'))
    #     model.save(os.path.join(tf_dir, 'conv_cout', f'conv_k3_i32_o{x}_hw56.tf'))
    #     quant_model(
    #         os.path.join(tf_dir, 'conv_cout', f'conv_k3_i32_o{x}_hw56.tf'),
    #         os.path.join(tflite_dir, 'conv_cout', 'fp32', f'conv_k3_i32_o{x}_hw56.tflite'),
    #         [1, 56, 56, 32]
    #     )

    # save dwconv
    # for x in [1, 3, 5, 7]:
    #     model = make_model([56, 56, 32], layers.DepthwiseConv2D(kernel_size=x, padding='same', depth_multiplier=1))
    #     model.save(os.path.join(tf_dir, 'dwconv_kernel', f'dwconv_k{x}_i32_o32_hw56.tf'))
    #     quant_model(
    #         os.path.join(tf_dir, 'dwconv_kernel', f'dwconv_k{x}_i32_o32_hw56.tf'),
    #         os.path.join(tflite_dir, 'dwconv_kernel', 'fp32', f'dwconv_k{x}_i32_o32_hw56.tflite'),
    #         [1, 56, 56, 32]
    #     )
    # for x in [1, 3, 5, 7]:
    #     model = make_model([14, 14, 320], layers.DepthwiseConv2D(kernel_size=x, padding='same', depth_multiplier=1))
    #     model.save(os.path.join(tf_dir, 'dwconv_kernel2', f'dwconv_k{x}_i320_o320_hw14.tf'))
    #     quant_model(
    #         os.path.join(tf_dir, 'dwconv_kernel2', f'dwconv_k{x}_i320_o320_hw14.tf'),
    #         os.path.join(tflite_dir, 'dwconv_kernel2', 'fp32', f'dwconv_k{x}_i320_o320_hw14.tflite'),
    #         [1, 14, 14, 320]
    #     )

    # for x in range(16, 49):
    #     model = make_model([56, 56, x], layers.DepthwiseConv2D(kernel_size=3, padding='same', depth_multiplier=1))
    #     model.save(os.path.join(tf_dir, 'dwconv_cio', f'dwconv_k3_io{x}_hw56.tf'))
    #     quant_model(
    #         os.path.join(tf_dir, 'dwconv_cio', f'dwconv_k3_io{x}_hw56.tf'),
    #         os.path.join(tflite_dir, 'dwconv_cio', 'fp32', f'dwconv_k3_io{x}_hw56.tflite'),
    #         [1, 56, 56, x]
    #     )


    # save relu
    # for x in range(16, 49):
    #     model = make_model([56, 56, x], layers.ReLU())
    #     model.save(os.path.join(tf_dir, 'relu', f'relu_hw56_c{x}.tf'))
    #     quant_model(
    #         os.path.join(tf_dir, 'relu', f'relu_hw56_c{x}.tf'),
    #         os.path.join(tflite_dir, 'relu', 'fp32', f'relu_hw56_c{x}.tflite'),
    #         [1, 56, 56, x]
    #     )

        
function_dict = {
    'fusion_test': fusion_test,
    'ncs2_test': ncs2_test,
    'prune_benchmark': prune_benchmark,
    'quant_op_test': quant_op_test
}

if __name__ == '__main__':
    assert len(sys.argv) > 1
    
    parser = argparse.ArgumentParser()
    parser.add_argument('func', type=str, help='specify the work to do')

    func = sys.argv[1]
    if func in function_dict.keys():
        function_dict[func](parser)
    else:
        raise ValueError(f'Function {func} not support. Supported functions: {list(function_dict.keys())}')