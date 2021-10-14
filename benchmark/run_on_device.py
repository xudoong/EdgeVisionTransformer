from torch.functional import norm
from .bench_utils import*
import os

def run_on_android(modelpath, adb, use_gpu=False, num_threads=1, num_runs=10, warmup_runs=10, skip_push=False, taskset_mask='10', benchmark_binary_dir='/data/tf_benchmark', no_root=False):
    if not skip_push:
        #=======Push to device===========
        adb.push_files(modelpath, '/sdcard/')
    model_name=modelpath.split('/')[-1]
    if benchmark_binary_dir[-1] == '/':
        benchmark_binary_dir = benchmark_binary_dir[:-1]
    benchmark_binary_path = f'{benchmark_binary_dir}/benchmark_model_plus_flex'

    command=f"taskset {taskset_mask} {benchmark_binary_path} --num_threads={num_threads} {'--use_gpu=true' if use_gpu else ''} --num_runs={num_runs} --warmup_runs={warmup_runs} --graph="+f'/sdcard/{model_name}'
    print(command)
    bench_str=adb.run_cmd(command, no_root=no_root)
    std_ms, avg_ms, mem_mb = fetech_tf_bench_results(bench_str)

    if not skip_push:
        #=======Clear device files=======
        adb.run_cmd(f'rm -rf /sdcard/{model_name}', no_root=no_root)
    return std_ms, avg_ms, mem_mb