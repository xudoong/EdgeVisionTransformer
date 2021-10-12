from .bench_utils import*
def run_on_android(modelpath, adb, use_gpu=False, num_threads=1, num_runs=10, warm_ups=10, skip_push=False):
    if not skip_push:
        #=======Push to device===========
        adb.push_files(modelpath, '/sdcard/')
    modelname=modelpath.split('/')[-1]


    command=f"taskset 10 /data/tf_benchmark/benchmark_model_plus_flex --num_threads={num_threads} {'--use_gpu=true' if use_gpu else ''} --num_runs={num_runs} --warmup_runs={warm_ups} --graph="+'/sdcard/'+modelname
    print(command)
    bench_str=adb.run_cmd(command)
    std_ms, avg_ms, mem_mb = fetech_tf_bench_results(bench_str)

    if not skip_push:
        #=======Clear device files=======
        adb.run_cmd(f'rm -rf /sdcard/'+modelname)
    return std_ms, avg_ms, mem_mb