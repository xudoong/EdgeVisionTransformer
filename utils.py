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


def onnx2tflite(onnx_model_path, output_path, save_tf=False):
    import os
    
    model_home = '/data/v-xudongwang/models'
    tf_prefix = os.path.join(model_home, 'tf_model')
    tflite_prefix = os.path.join(model_home, 'tflite_model')
    
    _, name = os.path.split(onnx_model_path)
    name = name.split('.')[0]

    tf_model_path = os.path.join(tf_prefix, name + '.tf')
    tflite_model_path = os.path.join(tflite_prefix, name + '.tflite') if output_path is None else output_path

    if os.path.exists(tf_model_path):
        os.system(f'rm -r {tf_model_path}')

    r = os.system(f'onnx-tf convert -i {onnx_model_path} -o {tf_model_path}')
    if r:
        exit(r)

    r = os.system(f'python /data/v-xudongwang/benchmark_tools/tools.py tf2tflite --input {tf_model_path} --output {tflite_model_path}')
    if r:
        if not save_tf:
            os.system(f'rm -r {tf_model_path}')
        exit(r)
    if not save_tf:
        os.system(f'rm -r {tf_model_path}')
    print('Convert successfully.')


def tf2tflite(saved_model_path, output_path, is_keras=False, is_keras_model=False):
    import tensorflow as tf
    # Convert the model
    if is_keras_model:
        model = saved_model_path
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    elif is_keras:
        model = tf.keras.models.load_model(saved_model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    else:
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path) # path to the SavedModel directory
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    tflite_model = converter.convert()

    # Save the model.
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f'Successfully convert model to {output_path}.')


def get_attention(h=768, a=12, h_k=None, is_tf=True, n=128):
    ''' Args:
    h: hidden_size
    a: num_attention_heads
    '''
    if is_tf:
        from modeling.layers.attention import Attention
        from modeling.layers.residual import Residual
        from modeling.layers.norm import LayerNorm

        attn = LayerNorm(Residual(Attention(h, a, h_k)))
        return attn
    else:
        from modeling.torch_layers.attention import Attention
        from modeling.torch_layers.norm import LayerNorm
        from modeling.torch_layers.residual import Residual
        attn = LayerNorm([n, h], Residual(Attention(h, a, h_k)))
        return attn


def get_ffn(h=768, i=3072, is_tf=True, n=128):
    '''Args:
    h: hidden_size
    i: intermediate_size
    '''
    if is_tf:
        from modeling.layers.residual import Residual
        from modeling.layers.norm import LayerNorm
        from modeling.layers.ffn import FeedForward

        ffn = LayerNorm(Residual(FeedForward(h, i)))
        return ffn
    
    else:
        from modeling.torch_layers.residual import Residual
        from modeling.torch_layers.norm import LayerNorm
        from modeling.torch_layers.ffn import FeedForward

        ffn = LayerNorm([n, h], Residual(FeedForward(h, i)))
        return ffn


def get_attention_plus_input(h=768, a=12, h_k=None, n=128, is_tf=True):
    if is_tf:
        import tensorflow as tf

        attn = get_attention(h, a, h_k)
        input = tf.keras.layers.Input(shape=[n, h], batch_size=1)
        output = attn(input)
        
        model = tf.keras.Model(input, output)
        return model
    else:
        attn = get_attention(h, a, h_k, is_tf=False, n=n)
        return attn


def get_ffn_plus_input(h=768, i=3072, n=128, is_tf=True):
    if is_tf:
        import tensorflow as tf

        ffn = get_ffn(h, i)
        input = tf.keras.layers.Input(shape=[n, h], batch_size=1)
        output = ffn(input)
        
        model = tf.keras.Model(input, output)
        return model
    else:
        ffn = get_ffn(h, i, is_tf=False, n=n)
        return ffn


def fetch_latency_std(file_path, begin_line=0, end_line=None):
    f = open(file_path)
    if end_line is None:
        lines = f.readlines()[begin_line:]
    else:
        lines = f.readlines()[begin_line: end_line]

    latency_list = []
    std_list = []

    for line in lines:
        if line.find('Avg latency') == -1: continue
        begin = line.find('Avg latency') + len('Avg latency ')
        end = line.find('ms') - 1
        latency = float(line[begin: end])

        begin = line.find('Std') + len('Std ')
        std = float(line[begin: -4])

        latency_list.append(latency)
        std_list.append(std)

    fmtL = "Q = " + ', '.join(["{:.2f}"]*len(latency_list))
    print(fmtL.format(*latency_list))
    print(fmtL.format(*std_list))


def get_onnx_model_inputs(model):
    import numpy as np
    inputs = {}
    for input in model.graph.input:
        name = input.name
        shape = []
        tensor_type = input.type.tensor_type
        for d in tensor_type.shape.dim:
            if d.HasField('dim_value'):
                shape.append(d.dim_value)
            else:
                shape.append(1)
        if len(shape) == 4:
            inputs[name] = np.random.randn(*shape).astype(np.float32)
        else:
            if 'mask' in name:
                inputs[name] = np.ones(shape=shape, dtype=np.int64)
            elif 'type' in name:
                inputs[name] = np.zeros(shape=shape, dtype=np.int64)
            else:
                inputs[name] = np.random.randint(low=0, high=10000, size=shape, dtype=np.int64)
    return inputs