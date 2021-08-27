import sys
import argparse
import timeit
import numpy as np


def test_onnx_latency():
    import onnx
    import onnxruntime as ort
    import torch 
    import numpy as np
    from utils import get_onnx_model_inputs

    parser = argparse.ArgumentParser()
    parser.add_argument('func', help='specify the work to do.')
    parser.add_argument('--model', required=True, type=str, help="onnx model path")
    
    parser.add_argument('--use_gpu', required=False, action='store_true', help="use GPU")
    parser.set_defaults(use_gpu=False)

    parser.add_argument('--num_runs',
                        required=False,
                        type=int,
                        default=10,
                        help="number of times to run per sample. By default, the value is 1000 / samples")

    args = parser.parse_args()

    execution_providers = ['CPUExecutionProvider'
                               ] if not args.use_gpu else ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(args.model, providers=execution_providers)
    model = onnx.load(args.model)

    latency_list = []
    for _ in range(args.num_runs):
        start_time = timeit.default_timer()
        session.run(None, get_onnx_model_inputs(model))
        latency = timeit.default_timer() - start_time
        latency_list.append(latency)

    avg_latency = np.average(latency_list)
    print(f'Avg latency: {avg_latency * 1000: .2f}ms')


def test_tf_latency():
    import tensorflow as tf
    parser = argparse.ArgumentParser()
    parser.add_argument('func', help='specify the work to do.')
    parser.add_argument('--model', required=True, type=str, help="SavedModel path")
    
    parser.add_argument('--use_gpu', required=False, action='store_true', help="use GPU")
    parser.set_defaults(use_gpu=False)

    parser.add_argument('--test_times',
                        required=False,
                        type=int,
                        default=5,
                        help="number of times to run per sample. By default, the value is 1000 / samples")
    parser.add_argument('--input_shape', required=True, type=str, help='input shape')

    args = parser.parse_args()

    input_shape = [int(num) for num in args.input_shape.split(',')]


    latency_list = []
    for _ in range(args.test_times):
        input_tensor = tf.random.uniform(input_shape)
        
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session() as sess:
                tf.saved_model.loader.load(
                    sess,
                    [tf.saved_model.tag_constants.SERVING],
                    args.model,
                )
                # output_placeholder = graph.get_tensor_by_name('StatefulPartitionedCall:0')
                output_placeholder = graph.get_tensor_by_name('PartitionedCall:0')
                
                input_placeholder = graph.get_tensor_by_name('serving_default_input:0')

                start_time = timeit.default_timer()

                sess.run(output_placeholder, feed_dict={
                    input_placeholder: np.random.uniform(0, 1, [1,3,224,224])
                })

                latency = timeit.default_timer() - start_time
                latency_list.append(latency)

    avg_latency = np.average(latency_list[1:])
    print(f'Avg latency: {avg_latency * 1000: .2f}ms')


def test_keras_latency():
    import tensorflow as tf
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument('func', help='specify the work to do.')
    parser.add_argument('--model', required=True, type=str, help="keras SavedModel path")
    
    parser.add_argument('--use_gpu', required=False, action='store_true', help="use GPU")
    parser.set_defaults(use_gpu=False)

    parser.add_argument('--test_times',
                        required=False,
                        type=int,
                        default=5,
                        help="number of times to run per sample. By default, the value is 1000 / samples")
    parser.add_argument('--input_shape', required=True, type=str, help='input shape')

    args = parser.parse_args()

    input_shape = [int(num) for num in args.input_shape.split(',')]


    model = tf.keras.models.load_model(args.model)
    print(f'Successfully loaded model from {args.model}.')

    latency_list = []
    for _ in range(args.test_times + 1):
        # input_tensor = np.random.uniform(0, 1, input_shape)
        inputs = None
        if isinstance(model.input, dict):
            inputs = {}
            for k, v in model.input.items():
               inputs[k] = tf.ones(shape=v.shape, dtype=v.dtype)
        else:
            inputs = tf.random.normal(input_shape)
        start_time = timeit.default_timer()

        _ = model(inputs)

        latency = timeit.default_timer() - start_time
        latency_list.append(latency)

    avg_latency = np.average(latency_list[1:])
    print(f'Avg latency: {avg_latency * 1000: .2f}ms')


def export_onnx_cmd():
    import torch
    from utils import export_onnx
    parser = argparse.ArgumentParser()
    parser.add_argument('func', help='specify the work to do.')
    parser.add_argument('--model', required=True, type=str, help="pytorch model path")
    parser.add_argument('--output', required=True, type=str, help='onnx output path')
    parser.add_argument('--input_shape', required=True, type=str, help='input shape')
    parser.add_argument('--opset_version', required=False, type=int, default=12, help='opset version')

    args = parser.parse_args()

    torch_model_path = args.model 
    onnx_model_path = args.output
    input_shape = [int(num) for num in args.input_shape.split(',')]
    opset_version = args.opset_version
    
    model = torch.load(torch_model_path)
    if isinstance(model, dict):
        if 'model' in model.keys():
            model = model['model']
        else:
            print('please specify the key to load model.')
            exit(-1)
    # model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
    export_onnx(model, onnx_model_path, input_shape, opset_version)


def export_onnx_deit():
    import torch
    from utils import export_onnx
    parser = argparse.ArgumentParser()
    parser.add_argument('func', help='specify the work to do.')
    parser.add_argument('--output', required=True, type=str, help='onnx output path')
    parser.add_argument('--input_shape', required=True, type=str, help='input shape')
    parser.add_argument('--type', type=str, choices=['tiny', 'small', 'base'], default='base', help='deit config')
    args = parser.parse_args()

    onnx_model_path = args.output
    input_shape = [int(num) for num in args.input_shape.split(',')]
    type = args.type

    import timm
    deit_type = f'deit_{type}_patch16_384'
    model = torch.hub.load('facebookresearch/deit:main', deit_type, pretrained=True)
    export_onnx(model, onnx_model_path, input_shape)


def export_onnx_bert_huggingface():
    import torch
    from utils import get_huggingface_bert

    parser = argparse.ArgumentParser()
    parser.add_argument('func', help='specify the work to do.')
    parser.add_argument('--layer', required=True, type=int, help='number of layers')
    parser.add_argument('--hidden', required=True, type=int, help='hidden size')
    parser.add_argument('--seq_len', required=False, default=128, type=int, help='sequence length')
    parser.add_argument('--output', required=True, type=str, help='output path')
    args = parser.parse_args()

    model = get_huggingface_bert(l=args.layer, h=args.hidden)
    seq_len = args.seq_len
    output_path = args.output

    inputs = {
        'input_ids': torch.randint(low=0, high=10000, size=[1, seq_len], dtype=torch.int64),
        'token_type_ids': torch.zeros(size=[1, seq_len], dtype=torch.int64),
        'attention_mask': torch.ones(size=[1, seq_len], dtype=torch.int64)
    }

    torch.onnx.export(
        model,
        tuple(inputs.values()),
        output_path,
        input_names=list(inputs.keys()),
        output_names=['last_hidden_state', 'pooler_output'],
        verbose=False,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        dynamic_axes={
            'input_ids': {0: 'batch_size'},
            'token_type_ids': {0: 'batch_size'},
            'attention_mask': {0: 'batch_size'},
            'last_hidden_state': {0: 'batch_size'},
            'pooler_output': {0: 'batch_size'},
        }
    )

    print(f'Successfully export bert huggingface model to {output_path}.')



def export_onnx_t2tvit():
    import torch 
    from utils import export_onnx 


    parser = argparse.ArgumentParser()
    parser.add_argument('func', help='specify the work to do.')
    parser.add_argument('--output', required=True, type=str, help='output path')
    parser.add_argument('--version', type=int, choices=[7, 10, 12, 14], required=True, help='T2T-ViT version')
    args = parser.parse_args()

    # import os, sys
    # t2tvit_path = '/data/v-xudongwang/other_codes/T2T-ViT-main'
    # sys.path.insert(0, t2tvit_path)
    from models import t2t_vit_7, t2t_vit_10, t2t_vit_12, t2t_vit_14

    if args.version == 7:
        model = t2t_vit_7()
    elif args.version == 10:
        model = t2t_vit_10()
    elif args.version == 12:
        model = t2t_vit_12()
    else:
        model = t2t_vit_14()

    export_onnx(model, args.output, [1,3,224,224])



def export_onnx_distilbert_huggingface():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument('func', help='specify the work to do.')
    parser.add_argument('--seq_len', required=False, default=128, type=int, help='sequence length')
    parser.add_argument('--output', required=True, type=str, help='output path')
    args = parser.parse_args()

    from transformers import DistilBertModel, DistilBertConfig
    config = DistilBertConfig()
    model = DistilBertModel(config)


    seq_len = args.seq_len
    output_path = args.output

    inputs = {
        'input_ids': torch.randint(low=0, high=10000, size=[1, seq_len], dtype=torch.int64),
        'attention_mask': torch.ones(size=[1, seq_len], dtype=torch.int64)
    }

    torch.onnx.export(
        model,
        tuple(inputs.values()),
        output_path,
        input_names=list(inputs.keys()),
        output_names=['last_hidden_state'],
        verbose=False,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        dynamic_axes={
            'input_ids': {0: 'batch_size'},
            'attention_mask': {0: 'batch_size'},
            'last_hidden_state': {0: 'batch_size'},
        }
    )

    print(f'Successfully export distilBERT huggingface model to {output_path}.')


def save_bert_encoder():
    from utils import get_bert_encoder
    parser = argparse.ArgumentParser()
    parser.add_argument('func', help='specify the work to do.')
    parser.add_argument('-l', required=True, type=int, help='number of layers')
    parser.add_argument('--hidden', required=True, type=int, help='hidden size')
    parser.add_argument('--seq_len', required=True, type=int, help='sequence length')
    parser.add_argument('--output', required=True, type=str, help='output path')
    args = parser.parse_args()

    model = get_bert_encoder(num_layers=args.l, hidden_size=args.hidden, num_heads=args.hidden//64, seq_len=args.seq_len)

    model.save(args.output)
    print(f'save model to {args.output}')


def export_onnx_proxyless_mobile():
    from utils import export_onnx
    parser = argparse.ArgumentParser()
    parser.add_argument('func', help='specify the work to do.')
    args = parser.parse_args()

    # export PYTHONPATH=/data/v-xudongwang/other_codes/proxylessnas-master
    from proxyless_nas import proxyless_mobile
    model = proxyless_mobile(pretrained=False)
    export_onnx(model, 'models/onnx_model/proxyless_mobile.onnx', [1,3,224,224])

def tf2tflite_cmd():
    import tensorflow as tf
    from utils import tf2tflite
    parser = argparse.ArgumentParser()
    parser.add_argument('func', help='specify the work to do.')
    parser.add_argument('--input', required=True, type=str, help='input path')
    parser.add_argument('--output', required=True, type=str, help='output path')
    parser.add_argument('--keras', dest='keras', action='store_true')
    parser.set_defaults(keras=False)
    args = parser.parse_args()

    saved_model_path = args.input
    output_path = args.output
    is_keras = args.keras
    
    tf2tflite(saved_model_path, output_path)



def mobile_benchmark():
    from benchmark.ADBConnect import ADBConnect
    from benchmark.run_on_device import run_on_android

    parser = argparse.ArgumentParser()
    parser.add_argument('func', help='specify the work to do.')
    parser.add_argument('--model', required=True, type=str, help='tflitemodel path')
    parser.add_argument('--use_gpu', dest='use_gpu', action='store_true')
    parser.add_argument('--num_runs', type=int, default=10, help='number of runs')
    parser.add_argument('--num_threads', type=int, default=1, help='number of threads')
    parser.set_defaults(use_gpu=False)
    args = parser.parse_args()

    model_path = args.model
    use_gpu = args.use_gpu
    num_threads = args.num_threads
    num_runs = args.num_runs
 
    adb = ADBConnect("98281FFAZ009SV")
    std_ms, avg_ms = run_on_android(model_path, adb, use_gpu,num_threads=num_threads, num_runs=num_runs)
    print(std_ms / avg_ms * 100, f'Avg latency {avg_ms} ms,', f'Std {std_ms} ms.')


def get_onnx_opset_version_cmd():
    from utils import get_onnx_opset_version

    parser = argparse.ArgumentParser()
    parser.add_argument('func', help='specify the work to do.')
    parser.add_argument('--model', required=True, type=str, help='tflitemodel path')
    args = parser.parse_args()

    onnx_model_path = args.model 
    
    opset_version = get_onnx_opset_version(onnx_model_path)
    print(opset_version)


def onnx2tflite_cmd():
    from utils import onnx2tflite
    parser = argparse.ArgumentParser()
    parser.add_argument('func', help='specify the work to do.')
    parser.add_argument('--model', required=True, type=str, help='onnx model path')
    parser.add_argument('--output', '-o', default=None, type=str, help='output tflite model path')
    parser.add_argument('--save_tf', action='store_true', dest='save_tf', help='to save tf SavedModel')
    parser.set_defaults(save_tf=False)
    args = parser.parse_args()

    onnx_model_path = args.model
    output_path = args.output
    save_tf = args.save_tf
    onnx2tflite(onnx_model_path, output_path, save_tf)


def save_vit():
    # first you need to : export PYTHONPATH=/data/v-xudongwang/other_codes/Vision-Transformer-main/
    from model import ViT
    import tensorflow as tf
    patch_size_list = [4, 8, 14, 16, 28, 32, 56]
    for patch_size in patch_size_list:
        vit_config = {"image_size":224,
                    "patch_size":patch_size,
                    "num_classes":1000,
                    "dim":768,
                    "depth":12,
                    "heads":12,
                    "mlp_dim":3072}

        vit = ViT(**vit_config)
        vit = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(3, vit_config["image_size"], vit_config["image_size"]), batch_size=1),
            vit,
        ])
        output_path = f'/data/v-xudongwang/models/tf_model/vit_patch{patch_size}_224.tf'
        vit.save(f'/data/v-xudongwang/models/tf_model/vit_patch{patch_size}_224.tf')
        print(f'Successfully save model to {output_path}.')


def get_flops_cmd():
    import tensorflow as tf
    from utils import get_flops
    parser = argparse.ArgumentParser()
    parser.add_argument('func', help='specify the work to do.')
    parser.add_argument('--model', required=True, type=str, help='keras model path')
    args = parser.parse_args()

    model_path = args.model 
    model = tf.keras.models.load_model(model_path)
    print('Flops: ', get_flops(model))


def export_onnx_mobilenet():
    import timm
    import torch
    from utils import export_onnx
    mobilenetv2 = timm.create_model('mobilenetv2_100', pretrained=True)
    export_onnx(mobilenetv2, 'models/onnx_model/mobilenetv2.onnx', input_shape=[1,3,224,224])
    mobilenetv3_large = timm.create_model('mobilenetv3_large_100', pretrained=True)
    export_onnx(mobilenetv3_large, 'models/onnx_model/mobilenetv3_large.onnx', input_shape=[1,3,224,224])


def export_onnx_vit_huggingface():
    from utils import get_huggingface_vit_model, export_onnx

    parser = argparse.ArgumentParser()
    parser.add_argument('func', help='specify the work to do.')
    parser.add_argument('--image_size', default=224, type=int, help='image_size')
    parser.add_argument('--patch_size', default=16, type=int, help='patch_size')
    parser.add_argument('--output', default=None, type=str, help='output onnx model path')
    args = parser.parse_args()

    image_size = args.image_size
    patch_size = args.patch_size
    output_path = args.output_path
    if output_path is None:
        output_path = f'models/onnx_model/vit_huggingface_patch{patch_size}_{image_size}.onnx'
    model = get_huggingface_vit_model(patch_size=patch_size, image_size=image_size)
    input_shape = [1, 3, image_size, image_size]
    export_onnx(model, output_path, input_shape)


def export_tflite_attention():
    from utils import tf2tflite, get_attention_plus_input

    parser = argparse.ArgumentParser()
    parser.add_argument('func', help='specify the work to do.')
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_heads', default=12, type=int)
    parser.add_argument('--head_size', default=None, type=int)
    parser.add_argument('--seq_len', default=128, type=int)
    parser.add_argument('--tf_path', default=None, type=str, help='tf savedModel path')
    parser.add_argument('--output', '-o', default=None, type=str, help='output tflite model path')
    args = parser.parse_args()

    h = args.hidden_size
    a = args.num_heads
    n = args.seq_len
    h_k = args.head_size
    tf_path = args.tf_path
    output_path = args.output
    if output_path is None:
        if h_k is None:
            output_path = f'models/tflite_model/attention_h{h}_a{a}_n{n}.tflite'
        else:
            output_path = f'models/tflite_model/attention_h{h}_a{a}_hk{h_k}_n{n}.tflite'

    attn = get_attention_plus_input(h, a, h_k, n)
    if tf_path:
        attn.save(tf_path)
    tf2tflite(attn, output_path, is_keras_model=True)


def export_tflite_ffn():
    from utils import tf2tflite, get_ffn_plus_input

    parser = argparse.ArgumentParser()
    parser.add_argument('func', help='specify the work to do.')
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--intermediate_size', '-i', default=3072, type=int)
    parser.add_argument('--seq_len', default=128, type=int)
    parser.add_argument('--tf_path', default=None, type=str, help='tf savedModel path')
    parser.add_argument('--output', '-o', default=None, type=str, help='output tflite model path')
    args = parser.parse_args()

    h = args.hidden_size
    i = args.intermediate_size
    n = args.seq_len
    tf_path = args.tf_path
    output_path = args.output
    if output_path is None:
        output_path = f'models/tflite_model/ffn_h{h}_i{i}_n{n}.tflite'
    
    attn = get_ffn_plus_input(h, i, n)
    if tf_path:
        attn.save(tf_path)
    tf2tflite(attn, output_path, is_keras_model=True)


def export_onnx_attention():
    from utils import get_attention_plus_input, export_onnx
    parser = argparse.ArgumentParser()
    parser.add_argument('func', help='specify the work to do.')
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_heads', default=12, type=int)
    parser.add_argument('--head_size', default=None, type=int)
    parser.add_argument('--seq_len', default=128, type=int)
    parser.add_argument('--output', '-o', default=None, type=str, help='output onnx model path')
    args = parser.parse_args()

    h = args.hidden_size
    a = args.num_heads
    n = args.seq_len
    h_k = args.head_size
    output_path = args.output
    if output_path is None:
        if h_k is None:
            output_path = f'models/onnx_model/attention_h{h}_a{a}_n{n}.onnx'
        else:
            output_path = f'models/onnx_model/attention_h{h}_a{a}_hk{h_k}_n{n}.onnx'

    model = get_attention_plus_input(h=h, a=a, h_k=h_k, n=n, is_tf=False)
    export_onnx(model, output_path, input_shape=[1, n, h])


def export_onnx_ffn():
    from utils import export_onnx, get_ffn_plus_input

    parser = argparse.ArgumentParser()
    parser.add_argument('func', help='specify the work to do.')
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--intermediate_size', '-i', default=3072, type=int)
    parser.add_argument('--seq_len', default=128, type=int)
    parser.add_argument('--output', '-o', default=None, type=str, help='output onnx model path')
    args = parser.parse_args()

    h = args.hidden_size
    i = args.intermediate_size
    n = args.seq_len
    output_path = args.output
    if output_path is None:
        output_path = f'models/onnx_model/ffn_h{h}_i{i}_n{n}.onnx'
    
    model = get_ffn_plus_input(h, i, n, is_tf=False)
    export_onnx(model, output_path, input_shape=[1, n, h])


def main():
    func = sys.argv[1]
    if func == 'test_onnx_latency':
        test_onnx_latency()
    elif func == 'export_onnx':
        export_onnx_cmd()
    elif func == 'export_onnx_deit':
        export_onnx_deit()
    elif func == 'save_bert_encoder':
        save_bert_encoder()
    elif func == 'tf2tflite':
        tf2tflite_cmd()
    elif func == 'mobile_benchmark':
        mobile_benchmark()
    elif func == 'get_onnx_opset_version':
        get_onnx_opset_version_cmd()
    elif func == 'test_tf_latency':
        test_tf_latency()
    elif func == 'test_keras_latency':
        test_keras_latency()
    elif func == 'export_onnx_bert_huggingface':
        export_onnx_bert_huggingface()
    elif func == 'onnx2tflite':
        onnx2tflite_cmd()
    elif func == 'export_onnx_t2tvit':
        export_onnx_t2tvit()
    elif func == 'export_onnx_distilbert_huggingface':
        export_onnx_distilbert_huggingface()
    elif func == 'save_vit':
        save_vit()
    elif func == 'get_flops':
        get_flops_cmd()
    elif func == 'export_onnx_mobilenet':
        export_onnx_mobilenet()
    elif func == 'export_onnx_proxyless_mobile':
        export_onnx_proxyless_mobile()
    elif func == 'export_onnx_vit_huggingface':
        export_onnx_vit_huggingface()
    elif func == 'export_tflite_attention':
        export_tflite_attention()
    elif func == 'export_tflite_ffn':
        export_tflite_ffn()
    elif func == 'export_onnx_attention':
        export_onnx_attention()
    elif func == 'export_onnx_ffn':
        export_onnx_ffn()

if __name__ == '__main__':
    main()