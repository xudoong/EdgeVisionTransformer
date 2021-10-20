from .bench_utils import*
import os

def run_on_android(modelpath, adb, use_gpu=False, num_threads=1, num_runs=10, warmup_runs=10, skip_push=False, 
                   taskset_mask='70', benchmark_binary_dir='/data/tf_benchmark', no_root=False, no_xnnpack=False, 
                   profiling_output_csv_file=None):
    if not skip_push:
        #=======Push to device===========
        adb.push_files(modelpath, '/sdcard/')
    model_name=modelpath.split('/')[-1]
    if benchmark_binary_dir[-1] == '/':
        benchmark_binary_dir = benchmark_binary_dir[:-1]
    benchmark_binary_path = f'{benchmark_binary_dir}/benchmark_model_plus_flex'

    command = f'taskset {taskset_mask} {benchmark_binary_path} --num_threads={num_threads} {"--use_gpu=true" if use_gpu else ""} '
    command += f'--num_runs={num_runs} --warmup_runs={warmup_runs} {"--use_xnnpack=false" if no_xnnpack else ""} --graph=/sdcard/{model_name} '
    command += f'--enable_op_profiling=true --profiling_output_csv_file=/sdcard/{os.path.basename(profiling_output_csv_file)} ' if profiling_output_csv_file else ''
    print(command)

    bench_str = adb.run_cmd(command, no_root=no_root)
    std_ms, avg_ms, mem_mb = fetech_tf_bench_results(bench_str)

    if not skip_push:
        #=======Clear device files=======
        adb.run_cmd(f'rm -rf /sdcard/{model_name}', no_root=no_root)

    if profiling_output_csv_file:
        adb.pull_files(src=f'/sdcard/{os.path.basename(profiling_output_csv_file)}', dst=profiling_output_csv_file)
        print(f'Save profiling output csv file in {profiling_output_csv_file}')
    return std_ms, avg_ms, mem_mb