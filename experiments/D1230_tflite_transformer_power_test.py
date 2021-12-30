import argparse
import re
import os
import sys
import subprocess
import re
from collections import defaultdict
import time 
import threading
import csv 

import numpy as np

sys.path.insert(0, f'{os.path.dirname(sys.argv[0])}/..')

RESULT_CSV_DIR = 'logs/D1230_transformer_power_test'
USB_POWER_THRESHOLD = 2300

class ADB:
    def __init__(self, serino):
        self.serino = serino
    
    def push(self, src, dst):
        subprocess.run(f'adb -s {self.serino} push {src} {dst}', shell=True)

    def pull(self, src, dst):
        subprocess.run(f'adb -s {self.serino} pull {src} {dst}', shell=True)

    def remove(self, dst):
        subprocess.run(f'adb -s {self.serino} shell rm {dst}', shell=True)

    def fetch_power(self):
        battery_current_str = self.run_cmd('taskset f cat /sys/class/power_supply/battery/current_now', mute=True)
        battery_voltage_str = self.run_cmd('taskset f cat /sys/class/power_supply/battery/voltage_now', mute=True)
        usb_current_str = self.run_cmd('taskset f cat /sys/class/power_supply/usb/input_current_now', mute=True)
        usb_voltage_str = self.run_cmd('taskset f cat /sys/class/power_supply/usb/voltage_now', mute=True)
        bc = float(battery_current_str) / 1e3 # <>e-6A / 1000 = <>mA
        bv = float(battery_voltage_str) / 1e6 # <>e-6V / 1e6 = <>V
        uc = float(usb_current_str) / 1e3
        uv = float(usb_voltage_str) / 1e6
        return bc, bv, uc, uv

    def run_cmd(self, cmd, mute=False):
        if not mute:
            print(cmd)
        result = subprocess.check_output(f'adb -s {self.serino} shell {cmd}', shell=True).decode('utf-8')
        if not mute:
            print(result)
        return result


stop_threads = False
def fetch_power(adb: ADB, delay: float, result_csv_path: str):
        try:
            f = open(result_csv_path, 'w')
            f.write('Battery Power(mW), USB Power(mW), Battery Current(mA), Battery Voltage(V), USB Current(mA), USB Voltage(V)\n')
            while True:
                # time.sleep(delay) # Force no delay
                bc, bv, uc, uv = adb.fetch_power()
                bp = bc * bv
                up = uc * uv
                f.write(', '.join([str(round(x, 2)) for x in [bp, up, bc, bv, uc, uv]]))
                f.write('\n')

                global stop_threads
                if stop_threads:
                    break
        finally:
            f.close()


class TfliteTester:
    def __init__(self, adb: ADB, model_zoo_dir: str):
        self.adb = adb
        self.model_zoo_dir = model_zoo_dir

    def _fetch_latency(self, text: str):
        match = re.findall(r'avg=[0-9e+.]+ ', text)[-1]
        return float(match[len('avg='): ]) / 1000

    def _get_result_csv_path(self, model_path: str):
        file_name = os.path.basename(model_path)
        return os.path.join(RESULT_CSV_DIR, file_name.replace('.tflite', '_power_result.csv'))

    def _benchmark_single(self, model_path):
        file_name = os.path.basename(model_path)
        dst_path = f'/sdcard/{file_name}'   
        avg_ms = 0.0
        self.adb.push(model_path, f'/sdcard/{file_name}')
        result_csv_path = self._get_result_csv_path(model_path)
        try:
            output_text_first = self.adb.run_cmd(
                f'taskset 70 /data/local/tmp/benchmark_model_plus_flex_r27 --graph={dst_path} --num_runs=5 --warmup_runs=5 --use_xnnpack=false --num_threads=1',
                mute=True)
            avg_ms_first = self._fetch_latency(output_text_first)
            num_runs_for_one_minute = int(1000 * 60 / avg_ms_first + 0.5)
            num_runs = max(num_runs_for_one_minute, 10)
            warmup_runs = num_runs // 2
            time.sleep(15)

            global stop_threads
            stop_threads = False
            power_fetcher = threading.Thread(target=fetch_power, args=(self.adb, 0.001, result_csv_path))
            power_fetcher.start()
            time.sleep(15)
            output_text = self.adb.run_cmd(
                f'taskset 70 /data/local/tmp/benchmark_model_plus_flex_r27 --graph={dst_path} --num_runs={num_runs} --warmup_runs={warmup_runs} --use_xnnpack=false --num_threads=1')
            time.sleep(15)
            stop_threads = True
            power_fetcher.join()
        except:
            pass
        
        self.adb.remove(dst_path)
        avg_ms = self._fetch_latency(output_text)
        return avg_ms

    def _get_battery_power(self, result_csv_path: str):
        battery_power_list = []
        with open(result_csv_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if len(row) > 1 and re.match(r'\d*\.?\d+', row[0]) and float(row[1]) > USB_POWER_THRESHOLD:
                    battery_power_list.append(float(row[0]))
        return np.average(battery_power_list)

    def _benchmark(self, ):
        print('===== Benchmarking =====')
        name_list = sorted(os.listdir(self.model_zoo_dir))
        result_dict = defaultdict(lambda: {})

        for model_name in name_list:
            model_path = os.path.join(self.model_zoo_dir, model_name)
            avg_ms = self._benchmark_single(model_path)
            result_dict[model_name]['avg_ms'] = round(avg_ms, 2)

            result_csv_path = self._get_result_csv_path(model_path)
            battery_power = self._get_battery_power(result_csv_path)
            result_dict[model_name]['battery_power'] = round(battery_power, 1)
        print('===============================')
        print('          SUMMARY')
        print('===============================')
        print(*name_list)
        for target in ['avg_ms', 'battery_power']:
            print(target, *[result_dict[k][target] for k in name_list])

    def run(self, ):
        self._benchmark()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_zoo_dir', default='models/tflite_model/project1_model_zoo_fp32', help='root dir to save tf and tflite models')
    parser.add_argument('--serial_number', default='98281FFAZ009SV', help='phone serial number')
    args = parser.parse_args()

    adb = ADB(args.serial_number)
    tester = TfliteTester(adb, args.model_zoo_dir)
    tester.run()

if __name__ == '__main__':
    main()