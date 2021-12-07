import tensorflow as tf
from .squeezenet import SqueezeNet
from . import shufflenet
from . import shufflenetv2
from .proxylessnas import get_proxylessnas
from .mnasnet import mnasnet_a1
import os


def add_keras_input_layer(model, input_shape, batch_size=1):
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape, batch_size=batch_size),
        model
    ])
    
def get_mobilenetv1():
    model = tf.keras.applications.MobileNet()
    return add_keras_input_layer(model, [224, 224, 3], 1)

def get_mobilenetv2():
    model = tf.keras.applications.MobileNetV2()
    return add_keras_input_layer(model, [224, 224, 3], 1)

def get_mobilenetv3small():
    model = tf.keras.applications.MobileNetV3Small()
    return add_keras_input_layer(model, [224, 224, 3], 1)

def get_mobilenetv3large():
    model = tf.keras.applications.MobileNetV3Large()
    return add_keras_input_layer(model, [224, 224, 3], 1)

def get_squeezenet():
    model = SqueezeNet(image_size=[224, 224, 3], batch_size=1)
    return model

def get_inception_resnetv2():
    model = tf.keras.applications.InceptionResNetV2() # input_shape=[299, 299, 3]
    return add_keras_input_layer(model, [299, 299, 3], 1) 

def get_inceptionv3():
    model = tf.keras.applications.InceptionV3() # input_shape=[299, 299, 3]
    return add_keras_input_layer(model, [299, 299, 3], 1)

def get_efficientnetb0():
    model = tf.keras.applications.EfficientNetB0()
    return add_keras_input_layer(model, [224, 224, 3], 1)

def get_resnet50():
    model = tf.keras.applications.ResNet50()
    return add_keras_input_layer(model, [224, 224, 3], 1)

def get_resnet50v2():
    model = tf.keras.applications.ResNet50V2()
    return add_keras_input_layer(model, [224, 224, 3], 1)

def get_shufflenet():
    model = shufflenet.shufflenet_g1_w1()
    return add_keras_input_layer(model, [224, 224, 3], 1)

def get_shufflenetv2():
    model = shufflenetv2.shufflenetv2_w1()
    return add_keras_input_layer(model, [224, 224, 3], 1)

def get_proxyless_mobile():
    model = get_proxylessnas('mobile')
    return add_keras_input_layer(model, [224, 224, 3], 1)

def get_mnasneta1():
    model = mnasnet_a1()
    return add_keras_input_layer(model, [224, 224, 3], 1)

cnn_zoo_dict = {
    'mobilenetv1': get_mobilenetv1,
    'mobilenetv2': get_mobilenetv2,
    'mobilenetv3small': get_mobilenetv3small,
    'mobilenetv3large': get_mobilenetv3large,
    'squeezenet': get_squeezenet,
    'inception_resnetv2': get_inception_resnetv2,
    'inceptionv3': get_inceptionv3,
    'efficientnetb0': get_efficientnetb0,
    'resnet50': get_resnet50,
    'resnet50v2': get_resnet50v2,
    'shufflenet': get_shufflenet,
    'shufflenetv2': get_shufflenetv2,
    'proxyless_mobile': get_proxyless_mobile, 
    'mnasneta1': get_mnasneta1
}

def get_model(model_name):
    if model_name not in cnn_zoo_dict.keys():
        raise ValueError(f'{model_name} not supported')
    return cnn_zoo_dict[model_name]()

def save_all(output_dir):
    for model_name in cnn_zoo_dict.keys():
        model = cnn_zoo_dict[model_name]()
        model.save(os.path.join(output_dir, model_name + '.tf'))