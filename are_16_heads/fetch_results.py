import argparse
import sys
from pathlib import Path
import os

def fetch_accuracy(parser: argparse.ArgumentParser):
    parser.add_argument('--file', '-f', type=str, help='input file to fecth accuracy')
    parser.add_argument('--begin_line', type=int, default=0, help='begin line number')
    parser.add_argument('--end_line'  , type=int, default=None, help='end line number plus one')
    args = parser.parse_args()
    
    f = open(args.file)
    if args.end_line:
        lines = f.readlines()[args.begin_line: args.end_line]
    else:
        lines = f.readlines()[args.begin_line:]

    i = 0
    acc_list = []
    for i in range(len(lines)):
        line = lines[i]
        if 'Pruning eval results' in line:
            tokens = lines[i + 1].split()
            acc = round(float(tokens[2]) * 100, 2)
            acc_list.append(acc)
        # if 'Finetuning eval results' in line:
        #     tokens = lines[i + 1].split()
        #     acc = round(float(tokens[3]) * 100, 2)
        #     acc_list.append(acc)
        i += 1

    print (acc_list)
    return acc_list


def fetch_accuracy_from_path(parser: argparse.ArgumentParser):
    parser.add_argument('--path', type=Path, help='pruned model directory to fetch accuracy')
    parser.add_argument('--finetuned', action='store_true', help='sub_dir is "final_finetuned" instead of "final"')
    args = parser.parse_args()

    final_str = 'final' if not args.finetuned else 'final_finetuned'

    model_list = sorted(os.listdir(args.path), key=lambda x: int(x[x.find('prune') + len('prune'): ]))
    acc_list = []
    for model_name in model_list:
        final_dir = args.path / model_name / final_str
        acc_file_name = sorted(os.listdir(final_dir))[0]
        if 'accuracy' not in acc_file_name:
            print(f'Please check the contents of this dir {final_dir}: {os.listdir(final_dir)}')
            exit()

        acc = round(int(acc_file_name[len('accuracy'): len('accuracy') + 4]) / 100, 2)
        acc_list.append(acc)
    
    print(acc_list)

function_dict = dict(
    fetch_accuracy = fetch_accuracy,
    fetch_accuracy_from_path = fetch_accuracy_from_path,
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('func', help='Specify the function to do.')
    
    assert(len(sys.argv) > 1)
    func = sys.argv[1]
    
    if func not in function_dict.keys():
        print('Supported functions: ', list(function_dict.keys()))
        exit()

    function_dict[func](parser)
