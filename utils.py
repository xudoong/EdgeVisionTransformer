def get_mobilenetv3large():
    import tensorflow as tf
    mobilenet = tf.keras.applications.MobileNetV3Large()
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=[224, 224, 3], batch_size=1),
        mobilenet
    ])
    model.save('/data/v-xudongwang/models/tf_model/mobilenetv3_large.tf')

def get_mobilenetv2():
    import tensorflow as tf
    mobilenet = tf.keras.applications.MobileNetV2()
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=[224, 224, 3], batch_size=1),
        mobilenet
    ])
    model.save('/data/v-xudongwang/models/tf_model/mobilenetv2.tf')


def get_huggingface_vit_model(patch_size=16, image_size=224):
    from transformers import ViTConfig, ViTModel
    config = ViTConfig(image_size=image_size, patch_size=patch_size)
    model = ViTModel(config)
    return model


def get_huggingface_mobile_bert():
    from transformers import MobileBertModel, MobileBertConfig
    # Initializing a MobileBERT configuration
    configuration = MobileBertConfig()
    # Initializing a model from the configuration above
    model = MobileBertModel(configuration)
    return model 


def get_huggingface_bert(l=12, h=768):
    from transformers import BertConfig, BertModel

    config = BertConfig(
        hidden_size=h,
        num_hidden_layers=l,
        num_attention_heads=h // 64,
        intermediate_size=h * 4,
    )
    model = BertModel(config)
    return model


def get_bert_encoder(num_layers, hidden_size, num_heads, seq_len, batch_size=None):
    import tensorflow as tf
    import tensorflow_hub as hub
    import tensorflow_text as text

    encoder = hub.KerasLayer(
    f"https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-{num_layers}_H-{hidden_size}_A-{num_heads}/2",
    trainable=True)

    input_word_ids = tf.keras.layers.Input(shape=[seq_len], batch_size=batch_size, name='input_word_ids',dtype=tf.int32)
    input_mask = tf.keras.layers.Input(shape=[seq_len], batch_size=batch_size, name='input_mask',dtype=tf.int32)
    input_type_ids = tf.keras.layers.Input(shape=[seq_len], batch_size=batch_size, name='input_type_ids',dtype=tf.int32)

    encoder_inputs = dict(
        input_word_ids = input_word_ids,
        input_mask = input_mask,
        input_type_ids = input_type_ids
    )
    encoder_output = encoder(encoder_inputs)
    model = tf.keras.Model(encoder_inputs, encoder_output)
    return model


def export_onnx(torch_model, output_path, input_shape, opset_version=12):
    import torch

    torch.onnx.export(
        torch_model,
        torch.randn(*input_shape),
        output_path,
        input_names=['input'],
        output_names=['output'],
        verbose=False,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                      'output' : {0 : 'batch_size'}}
    )

    print(f'Successfully export model to {output_path} as onnx.')


def get_flops(model):
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import  convert_variables_to_constants_v2_as_graph

    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec(shape=(1,) + model.input_shape[1:])])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        return flops.total_float_ops


def get_onnx_opset_version(onnx_model_path):
    import onnx
    model = onnx.load(onnx_model_path)
    return model.opset_import


def onnx2tflite(onnx_model_path):
    import os
    
    model_home = '/data/v-xudongwang/models'
    tf_prefix = os.path.join(model_home, 'tf_model')
    tflite_prefix = os.path.join(model_home, 'tflite_model')
    
    _, name = os.path.split(onnx_model_path)
    name = name.split('.')[0]

    tf_model_path = os.path.join(tf_prefix, name + '.tf')
    tflite_model_path = os.path.join(tflite_prefix, name + '.tflite')\

    if os.path.exists(tf_model_path):
        os.system(f'rm -r {tf_model_path}')

    r = os.system(f'onnx-tf convert -i {onnx_model_path} -o {tf_model_path}')
    if r:
        exit(r)
    r = os.system(f'python /data/v-xudongwang/benchmark_tools/tools.py tf2tflite --input {tf_model_path} --output {tflite_model_path}')
    if r:
        exit(r)
    print('Convert successfully.')