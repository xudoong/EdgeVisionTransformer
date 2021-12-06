import argparse
import subprocess
import os
import re
class ADB:
    def __init__(self, serino):
        self.serino = serino
    
    def push(self, src, dst):
        subprocess.check_output(f'adb -s {self.serino} push {src} {dst}', shell=True)

    def pull(self, src, dst):
        subprocess.check_output(f'adb -s {self.serino} pull {src} {dst}', shell=True)

    def remove(self, dst):
        subprocess.check_output(f'adb -s {self.serino} shell rm {dst}', shell=True)

    def run_cmd(self, cmd):
        result = subprocess.check_output(f'adb -s {self.serino} shell {cmd}', shell=True).decode('utf-8')
        return result

def fetch_number(text: str, marker: str):
    result = re.findall(f'{marker}\d+\.\d+', text)[0]
    return float(result[len(marker):])
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help='tflite model dir to test')
    parser.add_argument('--serino', default='98281FFAZ009SV', type=str, help='phone serial number to test')
    parser.add_argument('--precision', default=3, type=int, help='precision to print latency')
    args = parser.parse_args()

    adb = ADB(args.serino)
    name_list = []
    latency_list_f32 = []
    latency_list_f16 = []

    for name in sorted(os.listdir(args.model_dir)):
        f32_ms = 0
        f16_ms = 0
        try:
            name_list.append(os.path.splitext(os.path.basename(name))[0])
            model_path = os.path.join(args.model_dir, name)
            dst_path = f'/sdcard/{name}'
            adb.push(model_path, dst_path)
            result_f32 = adb.run_cmd(f'"cd /data/local/tmp && ./benchmark_model_fixed_group_size --graph={dst_path} --use_gpu=true --precision=F32"')
            result_f16 = adb.run_cmd(f'"cd /data/local/tmp && ./benchmark_model_fixed_group_size --graph={dst_path} --use_gpu=true --precision=F16"')
            adb.remove(dst_path)
            f32_ms = fetch_number(result_f32, 'comp_avg_ms=')
            f16_ms = fetch_number(result_f16, 'comp_avg_ms=')
        except:
            pass
        latency_list_f32.append(round(f32_ms, args.precision))
        latency_list_f16.append(round(f16_ms, args.precision))
        
        print(name_list[-1], f32_ms, f16_ms)

    print('==== LATENCY SUMMARY ====')
    print(name_list)
    print('[F32 Latency]')
    print(latency_list_f32)
    print('[F16 Latency]')
    print(latency_list_f16)


if __name__ == '__main__':
    main()