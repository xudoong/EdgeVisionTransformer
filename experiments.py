import tensorflow as tf
import tensorflow.keras.layers as layers
import os
import sys
import argparse

from modeling.layers.activation import gelu
from utils import add_keras_input_layer


class FusionTest():
    def __init__(self, output_tf_dir, output_tflite_dir, l=197, h=768, i=3072) -> None:
        self.output_tf_dir = output_tf_dir
        self.output_tflite_dir = output_tflite_dir
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


def fusion_test(parser: argparse.ArgumentParser):
    parser.add_argument('--output_tf_dir', required=True, type=str, help='output tf saved model dir')
    parser.add_argument('--output_tflite_dir', required=True, type=str, help='output tflite model dir')
    parser.add_argument('--type', choices=['tiny', 'base'], help='this argument determines the op size')
    args = parser.parse_args()
    
    if args.type == 'base':
        l, h, i = 197, 768, 768 * 4
    else:
        l, h, i = 197, 192, 192 * 4

    fusion_test_class = FusionTest(args.output_tf_dir, args.output_tflite_dir, l, h, i)
    fusion_test_class.save_model()
    fusion_test_class.convert_to_tflite()


function_dict = {
    'fusion_test': fusion_test,
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