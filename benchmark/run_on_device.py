from .bench_utils import*
def run_on_android(modelpath, adb, use_gpu=False, num_threads=1, num_runs=10):
    #=======Push to device===========
    adb.push_files(modelpath, '/sdcard/')
    modelname=modelpath.split('/')[-1]


    command=f"taskset 70 /data/tf_benchmark/benchmark_model_plus_flex --num_threads={num_threads} {'--use_gpu=true' if use_gpu else ''} --warm_ups=10 --num_runs={num_runs} --graph="+'/sdcard/'+modelname
    print(command)
    bench_str=adb.run_cmd(command)
    std_ms,avg_ms=fetech_tf_bench_results(bench_str)

    #=======Clear device files=======
    adb.run_cmd(f'rm -rf /sdcard/'+modelname)
    return std_ms,avg_ms