import os
import sys
import argparse
import csv
import re
import subprocess


def get_root_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('func', help='specify the work to do.')
    parser.add_argument('--openvino_root', default=None, help='openvino root directory')
    return parser

def process_root_args(args):
    openvino_root = args.openvino_root
    if args.openvino_root is None:
        if os.environ.get('openvino') is None:
            exit('Openvino root dir must be specified or openvino env var must exists.')
        openvino_root = os.environ.get('openvino')

    mo_path = os.path.join(openvino_root, 'deployment_tools', 'model_optimizer', 'mo.py')
    benchmark_app_path = os.path.join(openvino_root, 'deployment_tools', 'tools', 'benchmark_tool', 'benchmark_app.py')
    return dict(
        openvino_root_path=openvino_root,
        mo_path=mo_path,
        benchmark_app_path=benchmark_app_path
    )


def model_optimize(mo_path, input_path, output_dir, batch_size=None, input_shape=None, data_type='FP32'):
    cmd = f'python "{mo_path}" --input_model "{input_path}" --output_dir "{output_dir}" --data_type {data_type}'
    if batch_size:
        cmd += f' --batch={batch_size}'
    if input_shape:
        cmd += f' --input_shape="{input_shape}"'

    print(cmd)
    os.system(cmd)


def model_optimize_cli():
    parser = get_root_parser()
    parser.add_argument('--input_path', '-i', required=True, type=str, help='model_path or the directory of models to convert.')
    parser.add_argument('--output_dir', '-o', default='models/vino_model', type=str, help='output ir path')
    parser.add_argument('--batch_size', '-b', default=None, type=int, help='input tensor batch size')
    parser.add_argument('--input_shape', default=None, type=str, help='input tensor shape')
    parser.add_argument('--data_type', default='FP32', type=str, help='data type for quantization')
    args = parser.parse_args()
    path_dict = process_root_args(args)

    input_path = args.input_path
    output_dir = args.output_dir
    mo_path = path_dict['mo_path']
    batch_size = args.batch_size
    data_type = args.data_type
    input_shape = None
    if args.input_shape:
        input_shape = [int(x) for x in args.input_shape.split(',')]
    is_dir = os.path.isdir(input_path)

    input_list = []
    if not is_dir:
        input_list.append(input_path)
    else:
        for name in sorted(os.listdir(input_path)):
            input_list.append(os.path.join(input_path, name))

    for model_path in input_list:
        model_optimize(mo_path, model_path, output_dir, batch_size, input_shape=input_shape, data_type=data_type)


def openvino_benchmark(benchmark_app_path, model_path, niter=10, num_threads=1, batch_size=1, device='CPU', layername_pattern=r'Total', layertype_pattern=None, csv_output_dir=None, show_detail=True):
    # setup_envs_path = os.path.join(openvino_root_path, 'bin')
    # source_cmd = f'"{os.path.join(setup_envs_path, os.listdir(setup_envs_path)[0])}"'
    benchmark_cmd = f'python "{benchmark_app_path}" -m "{model_path}" -niter={niter} -nthreads={num_threads} -b={batch_size} -d {device} -nireq=1 -api=sync --report_type=detailed_counters'
    # cmd = source_cmd + ' && ' + benchmark_cmd
    print(benchmark_cmd)
    subprocess.run(benchmark_cmd, shell=True)

    if show_detail:
        if os.name == 'nt':
            os.system('type benchmark_detailed_counters_report.csv')
        else:
            os.system('cat benchmark_detailed_counters_report.csv')

    latency = 0.0
    with open('benchmark_detailed_counters_report.csv', 'r') as f:
        csvreader = csv.reader(f, delimiter=';')
        for r in csvreader:
            if len(r) and re.fullmatch(layername_pattern, r[0]) and (layertype_pattern is None or re.fullmatch(layertype_pattern, r[2])):
                latency += float(r[4])

    if csv_output_dir:
        subprocess.run(f'mv benchmark_detailed_counters_report.csv {csv_output_dir}', shell=True)
        subprocess.run(f'mv benchmark_report.csv {csv_output_dir}', shell=True)
    else:
        os.remove('benchmark_detailed_counters_report.csv')
        os.remove('benchmark_report.csv')
    
    return latency       


def openvino_benchmark_cli():
    parser = get_root_parser()
    parser.add_argument('--model_path', '--model', '-m', required=True, type=str, help='xml openvino IR model path')
    parser.add_argument('--num_runs', '-niter', default=50, type=int, help='num of runs')
    parser.add_argument('--num_threads', default=1, type=int, help='num of threads')
    parser.add_argument('--batch_size', '-b', default=1, type=int, help='num of threads')
    parser.add_argument('--device', '-d', default='CPU', type=str, help='device list to benchmark on')
    args = parser.parse_args()
    
    path_dict = process_root_args(args)
    benchmark_app_path = path_dict['benchmark_app_path']
    openvino_root_path = path_dict['openvino_root_path']
    num_runs = args.num_runs
    num_threads = args.num_threads
    batch_size = args.batch_size
    device = args.device
    is_dir = os.path.isdir(args.model_path)

    model_list = []
    if not is_dir:
        model_list.append(args.model_path)
    else:
        for name in sorted(os.listdir(args.model_path)):
            if '.xml' in name:
                model_list.append(os.path.join(args.model_path, name))

    print('[ INFO ] Models to benchmark:')
    for name in model_list:
        print(os.path.basename(name), end='\t')
    print('')
    latency_list = []
    latency_dict = {}
    for model_path in model_list:
        latency = openvino_benchmark(openvino_root_path, benchmark_app_path, model_path, niter=num_runs, num_threads=num_threads, batch_size=batch_size, device=device)
        latency_list.append(latency)
        latency_dict[os.path.basename(model_path)] = latency

    print('[ BENCHMARK FINISHED ]')
    print('')
    print('[ SUMMARY ]')
    print(latency_dict)
    print(latency_list)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        exit('Must specify the function in argv[1]')

    func = sys.argv[1]

    if func in  ['mo', 'model_optimize']:
        model_optimize_cli()
    
    if func in ['benchmark']:
        openvino_benchmark_cli()
