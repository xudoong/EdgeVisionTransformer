# EdgeVisionTransformer
Tools to measure the inference performance of vision transformers on mobile devices, plus pruning methods adopted from previous work: [are16heads](https://github.com/pmichel31415/are-16-heads-really-better-than-1) , [nn_pruning](https://github.com/huggingface/nn_pruning).

The folder *model_zoo* contains the tflite models we tested.

The folder *modeling.models* contains our tensorflow implemented ViT (DeiT) and T2T-ViT model. Thanks for the work of https://github.com/kamalkraj/Vision-Transformer and https://github.com/yitu-opensource/T2T-ViT. The tensorflow swin-transformer is from https://github.com/rishigami/Swin-Transformer-TF.git.

# Usage
First clone this repo and install dependencies.
```Bash
git clone https://github.com/RendezvousHorizon/EdgeVisionTransformer
pip install -r requirements.txt

# to prune deit
pip install -r deit_pruning/requirements.txt
```

### Benchmark models on mobile devices

File tools.py provides the command line interface to export, convert and benchmark cnn and transformer models. Example usage includes:

#### 1) convert tensorflow keras saved model to tflite

```python
python tools.py tf2tflite --input <saved_model_path> --output <tflite_model_path> [--quantization=float16|dynamic]
```

You can add quantization argument to quantize the model when converting. We tried two quantization methods from tensorflow: [dynamic range quantization](https://www.tensorflow.org/lite/performance/post_training_quant) and [float16 quantization](https://www.tensorflow.org/lite/performance/post_training_float16_quant).

#### 2) benchmark tflite model inference latency and memory on mobile phones

First you need to setup adb and plug a android phone to your computer.

Next download or compile [tflite_benchmark_binary](https://www.tensorflow.org/lite/performance/measurement#native_benchmark_binary). We use the nightly pre-built binary that support TF ops via Flex delegate provided by tensorflow official website ([download link](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_benchmark_model_plus_flex)). If you build by yourself, be sure to build with Tensorflow ops support because all ops in transformer are not supported by tflite (e.g. Einsum, ExtractImagePatches, Erf, Roll).

Then you can push the binary and tflite models to your phone and run benchmark.

```bash
adb -s <SERIAL_NO> push benchmark_model_plus_flex /data/local/tmp
adb -s <SERIAL_NO> push model.tflite /sdcard

# run benchmark by logging in to the phone
abs -s <SERIAL_NO> shell 
chmod +x /data/local/tmp/benchmark_model_plus_flex
taskset <core_mask> /data/local/tmp/benchmark_model_plus_flex --graph=/sdcard/model.tflite --num_runs=50 --warmup_runs=50 --num_threads=1 [--enable_op_profiling --profiling_output_csv_file=profile_output.csv]
# or run with tools.py interface
python tools.py mobile_benchmark  --serial_number <SERIAL_NO> --model model.tflite --taskset_mask <core_mask> --benchmark_binary_dir /data/local/tmp --num_runs=50 --warmup_runs=50 [--profiling_output_csv_file output.csv]

```

The benchmark output shows the average inference latency and delta memory footprint between the beginning and end of the inference:

```bash
# benchmark mobilenet_v2
The input model file size (MB): 13.9926
Initialized session in 0.847ms.
Running benchmark for at least 30 iterations and at least 0.5 seconds but terminate if exceeding 150 seconds.
count=30 first=57338 curr=46893 min=45730 max=57338 avg=47674.1 std=2073

Running benchmark for at least 30 iterations and at least 1 seconds but terminate if exceeding 150 seconds.
count=30 first=47632 curr=49309 min=44914 max=49309 avg=47440.3 std=1087

Inference timings in us: Init: 847, First inference: 57338, Warmup (avg): 47674.1, Inference (avg): 47440.3
Note: as the benchmark tool itself affects memory footprint, the following is only APPROXIMATE to the actual memory footprint of the model at runtime. Take the information at your discretion.
Peak memory footprint (MB): init=1.26562 overall=26.6641
```

### Prune DeiT

The folder *deit_pruning* contains code adopted from [nn_pruning](https://github.com/huggingface/nn_pruning) to structure prune DeiT.

Example usage are as follows:

#### 1) prune DeiT

```bash
cd deit_pruning
python -m torch.distributed.launch --nproc_per_node 4 src/train_main.py --deit_model_name facebook/deit-tiny-patch16-224 --output_dir <output_dir> --data_path <imagenet2012_dataset_path> --sparse_preset topk-hybrid-struct-layerwise-tiny --layerwise_thresholds h_0.50_d_0.3-h_0.50_d_0.3-h_0.50_d_0.3-h_0.50_d_0.3-h_0.50_d_0.3-h_0.50_d_0.3-h_0.50_d_0.3-h_0.50_d_0.3-h_0.50_d_0.3-h_0.50_d_0.3-h_0.50_d_0.3-h_0.50_d_0.3 --nn_pruning --do_eval --micro_batch_size 256 --scale_lr --epoch 6 

```

The code will load the deit_model using function`transformers.AutoModelForImageClassification.from_pretrained()` with path specified by argument`--deit_model_name`.  `--data_path ` is the path to imagenet2012 dataset, which must contain two sub directories *train* and *val*. `--layerwise_thresholds` specifies the threshold per layer to prune attention heads and FFN. For example, for DeiT-Tiny with 3 heads per layer, `h_0.668_d_0.9` means prune 1 head and 10% FFN rows/columns. 

The implementation of nn_pruning could not ideally support prune attention heads: it prunes q, k, v weight matrixes separately. We adopt it to prune heads with score set to be the total number of q, k, v matrixes not pruned. If the q, k, v matrixes of a head are all pruned, then its score is zero and we prune it first. If the q, k, v matrixes are all not pruned, then the head's score is 3 and we prune it last.

As a result, there are q, k, v blocks completely zero after prune heads, so finetune is necessary. 

#### 2) finetune pruned DeiT

```bash
cd deit_pruning
python -m torch.distributed.launch --nproc_per_node 4 src/train_main.py --deit_model_name <pruned_deit_dir> --output_dir <output_dir> --data_path <imagenet2012_dataset_path> --final_finetune --micro_batch_size 256 --scale_lr --epoch 3
```



Additionally, during experiment, we found kd (knowledge distillation) could help improve accuracy a lot. You can add kd into pruning or finetuning  by appending the following arguments:

```bash
--do_distil --alpha_distil=0.9 --teacher_model=facebook:deit-base-patch16-224 
```



