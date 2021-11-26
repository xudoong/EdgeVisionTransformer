#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys
import argparse
import numpy as np
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit
import tensorrt as trt
import onnx
import torch

import common
from calibrator import DummyCalibrator


# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def get_onnx_input_shape(model_path):
    model = onnx.load(model_path)
    input0 = model.graph.input[0]
    tensor_type = input0.type.tensor_type
    input_shape = []
    for d in tensor_type.shape.dim:
        if d.HasField('dim_value'):
            input_shape.append(d.dim_value)
        else:
            input_shape.append(1)
    return input_shape

# The Onnx path is used for Onnx models.
def build_engine_onnx(model_file, quant=None):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(common.EXPLICIT_BATCH)
    config = builder.create_builder_config()
    runtime = trt.Runtime(TRT_LOGGER)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config.max_workspace_size = common.GiB(1)

    if quant == 'int8' or quant == 'both':
        config.set_flag(trt.BuilderFlag.INT8)
        input_shape = get_onnx_input_shape(model_file)
        dummy_input = torch.rand(input_shape).numpy()
        config.int8_calibrator = DummyCalibrator(dummy_input, batch_size=1)
    if quant == 'fp16' or quant == 'both':
        config.set_flag(trt.BuilderFlag.FP16)

    # Load the Onnx model and parse it in order to populate the TensorRT network.
    with open(model_file, 'rb') as model:
        if not parser.parse(model.read()):
            print ('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print (parser.get_error(error))
            return None

    plan = builder.build_serialized_network(network, config)
    return runtime.deserialize_cuda_engine(plan)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str, help="torch state_dict path")
    parser.add_argument('--quant', default=None, choices=['int8', 'fp16', 'both'], help='inference with int8')
    parser.add_argument('--num_runs', default=50, type=int, help='number of inference runs')
    parser.add_argument('--warmup_runs', default=20, type=int, help='number of warmup runs')
    parser.add_argument('--topk', default=None, type=int, help='take the avg of top k latency to reduce variance')
    parser.add_argument('--precision', default=3, type=int, help='the precision of latency result')
    args = parser.parse_args()


    # Build a TensorRT engine.
    engine = build_engine_onnx(args.model, quant=args.quant)
    # Inference is the same regardless of which parser is used to build the engine, since the model architecture is the same.
    # Allocate buffers and create a CUDA stream.
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    # Contexts are used to perform inference.
    context = engine.create_execution_context()

    latency_list = []
    for _ in range(10):
        common.do_inference_v2_with_timer(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    for _ in range(50):
        latency_ms = common.do_inference_v2_with_timer(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        latency_list.append(latency_ms)

    if args.topk:
        latency_list.sort()
        latency_list = latency_list[:args.topk]

    avg_ms = np.average(latency_list)
    std_ms = np.std(latency_list)
    print(f'{os.path.basename(args.model)}    Avg latency {avg_ms:.3f} ms    Std {std_ms:.3f} ms')


if __name__ == '__main__':
    main()
