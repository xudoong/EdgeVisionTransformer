import argparse
import enum
import re
import sys
from types import prepare_class

'''--------------------------------------------------------------
Tools to analyse tflite benchmark_model profiling output csv file.
--------------------------------------------------------------'''

def _replace_flex(name: str, type: str):
    op = name.split(':')[0].split('/')[-1]
    op = op.lower()
    if type == 'swin':
        if 'transpose' in op: return 'TRANSPOSE'
        if 'add' in op: return 'ADDv2'
        if 'roll' in op: return 'ROLL'
        if 'erf' in op: return 'ERF'
    if type == 't2t_vit':
        if 'einsum' in op: return 'EINSUM'
        if 'extractimagepatches' in op: return 'EXTRACTIMAGEPATCHES'
    return 'TFFLEXDELEGATE'


def _find_op_wise_line_range(rows):
    schema = {}
    for begin_line in range(len(rows)):
        row = rows[begin_line]
        if len(row) == 1 and 'Operator-wise Profiling Info for Regular Benchmark Run' in row[0]:
            schema_row = rows[begin_line + 2]
            schema = {schema_row[i].strip(): i for i in range(len(schema_row))}
            begin_line += 3
            break
    end_line = begin_line
    while True:
        if len(rows[end_line]) < len(schema):
            break
        end_line += 1
    return begin_line, end_line, schema


def _read_rows(file_path):
    import csv
    rows = []
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            rows.append(row)
    return rows


def analyse_op(parser: argparse.ArgumentParser):
    parser.add_argument('--file', type=str, required=True, help='csv profile result file')
    parser.add_argument('--type', choices=['swin', 't2t_vit'], required=True, help='transformer model type')
    args = parser.parse_args()

    rows = _read_rows(args.file)
    begin_line, end_line, schema = _find_op_wise_line_range(rows)
    print(f'Schema: {schema}')

    result_table = {}
    for row in rows[begin_line: end_line]:
        node_type = row[schema['node type']]
        if 'TfLiteFlexDelegate' in node_type:
            node_type = _replace_flex(row[schema['name']], args.type)
        if node_type in result_table.keys():
            result_table[node_type]['latency'] += float(row[schema['avg_ms']])
            result_table[node_type]['percent'] += float(row[schema['%']][:-1])
        else:
            result_table[node_type] = {}
            result_table[node_type]['latency'] = float(row[schema['avg_ms']])
            result_table[node_type]['percent'] = float(row[schema['%']][:-1])

    for k, v in result_table.items():
        print(f'{k} {v["latency"]: .2f} {v["percent"]: .2f}')


def analyse_gelu_ln(parser: argparse.ArgumentParser):
    import csv
    parser.add_argument('--file', type=str, required=True, help='csv profile result file')
    parser.add_argument('--type', choices=['deit', 'swin', 't2t_vit'], required=True, help='the transformer model type')
    args = parser.parse_args()

    rows = _read_rows(args.file)
    begin_line, end_line, schema = _find_op_wise_line_range(rows)
    print(f'Schema: {schema}')

    gelu_latency = 0
    gelu_percent = 0
    ln_latency = 0
    ln_percent = 0

    hit_gelu = 0
    hit_ln = 0
    for i in range(begin_line, end_line):
        row = rows[i]
        node_type = row[schema['node type']]

        if args.type == 'deit':
            if 'POW' in node_type:
                hit_gelu += 1
                for j in range(8):
                    gelu_latency += float(rows[i + j][schema['avg_ms']])
                    gelu_percent += float(rows[i + j][schema['%']][:-1])
            if 'FULLY_CONNECTED' not in node_type and 'RESHAPE' not in node_type and 'layer_normalization' in rows[i][schema['name']]:
                hit_ln += 1
                ln_latency += float(rows[i][schema['avg_ms']])
                ln_percent += float(rows[i][schema['%']][:-1])

        elif args.type == 'swin':
            if 'gelu' in rows[i][schema['name']].lower():
                hit_gelu += 1
                gelu_latency += float(rows[i][schema['avg_ms']])
                gelu_percent += float(rows[i][schema['%']][:-1])
            if 'norm' in rows[i][schema['name']].lower():
                hit_ln += 1
                ln_latency += float(rows[i][schema['avg_ms']])
                ln_percent += float(rows[i][schema['%']][:-1])

        elif args.type == 't2t_vit':
            if 'POW' in node_type:
                hit_gelu += 1
                for j in range(8):
                    gelu_latency += float(rows[i + j][schema['avg_ms']])
                    gelu_percent += float(rows[i + j][schema['%']][:-1])
            if 'layer_normalization' in rows[i][schema['name']].lower():
                hit_ln += 1
                ln_latency += float(rows[i][schema['avg_ms']])
                ln_percent += float(rows[i][schema['%']][:-1])

    
    print('hit_gelu {} hit_ln {} gelu_latency {:.2f} gelu_percent {:.2f} ln_latency {:.2f} ln_percent {:.2f}'.format(
        hit_gelu, hit_ln, gelu_latency, gelu_percent, ln_latency, ln_percent))


def analyse_attn_ffn(parser: argparse.ArgumentParser):
    import csv
    parser.add_argument('--file', type=str, required=True, help='csv profile result file')
    parser.add_argument('--type', choices=['deit', 'swin', 't2t_vit'], required=True, help='the transformer model type')
    args = parser.parse_args()

    rows = _read_rows(args.file)
    begin_line, end_line, schema = _find_op_wise_line_range(rows)
    rows = sorted(rows[begin_line: end_line], key=lambda row: float(row[schema['start']]))
    print(f'Schema: {schema}')
    attn_percent = 0
    attn_latency = 0
    ffn_percent = 0
    ffn_latency = 0
    pre_post_processing_percent = 0
    pre_post_processing_latency = 0

    if args.type == 'deit' or args.type == 't2t_vit':
        pre_ln_str = 'Null'
        is_ffn = 1
        for row in rows:
            if 'transformer_encoder_block' not in row[schema['name']]:
                pre_post_processing_latency += float(row[schema['avg_ms']])
                pre_post_processing_percent += float(row[schema['%']][:-1])
            else:
                ln_str = re.match(r'.*/(layer_norm_?\d*)/.*', row[schema['name']]).groups()[0]
                if ln_str != pre_ln_str:
                    pre_ln_str = ln_str
                    is_ffn = (is_ffn + 1) % 2
                if is_ffn == 0:
                    attn_latency += float(row[schema['avg_ms']])
                    attn_percent += float(row[schema['%']][:-1])
                else:
                    ffn_latency += float(row[schema['avg_ms']])
                    ffn_percent += float(row[schema['%']][:-1])
    else: # swin
        for row in rows:
            if 'swin_transformer_block' not in row[schema['name']]:
                pre_post_processing_latency += float(row[schema['avg_ms']])
                pre_post_processing_percent += float(row[schema['%']][:-1])
            elif 'window_attention' in row[schema['name']] or 'norm1' in row[schema['name']]:
                attn_latency += float(row[schema['avg_ms']])
                attn_percent += float(row[schema['%']][:-1])
            elif 'mlp' in row[schema['name']] or 'norm2' in row[schema['name']]:
                ffn_latency += float(row[schema['avg_ms']])
                ffn_percent += float(row[schema['%']][:-1])
            elif 'norm' not in row[schema['name']]:
                pre_post_processing_latency += float(row[schema['avg_ms']])
                pre_post_processing_percent += float(row[schema['%']][:-1])
            else:
                raise RuntimeError()

    print(f'{args.type} | attn (percent, latency) = ({attn_percent:.2f}, {attn_latency:.2f}) | ' + 
          f'ffn (percent, latency) = ({ffn_percent:.2f}, {ffn_latency:.2f}) | ' +
          f'pre & post-processing (percent, latency) = ({pre_post_processing_percent:.2f}, {pre_post_processing_latency:.2f})')


def fetch_all_op_latency(parser: argparse.ArgumentParser):
    import csv
    parser.add_argument('--file', type=str, required=True, help='csv profile result file')
    parser.add_argument('--op', choices=['conv', 'dwconv', 'dense'], required=True, help='op type to fetch latency')
    args = parser.parse_args()

    OP_NAME_DICT = {
        'conv': 'CONV_2D',
        'dwconv': 'DEPTHWISE_CONV_2D',
        'dense': 'FULLY_CONNECTED'
    }

    rows = _read_rows(args.file)
    begin_line, end_line, schema = _find_op_wise_line_range(rows)
    latency_list = []
    rows = sorted(rows[begin_line: end_line], key=lambda row: float(row[schema['start']]))

    for row in rows:
        node_type = row[schema['node type']]
        if node_type == OP_NAME_DICT[args.op]:
            latency_list.append(round(float(row[schema['avg_ms']]), 2))

    print(f'{args.op} count = {len(latency_list)}')
    print(latency_list)


function_dict = {
    'analyse_op': analyse_op,
    'analyse_gelu_ln': analyse_gelu_ln,
    'analyse_attn_ffn': analyse_attn_ffn,
    'fetch_all_op_latency': fetch_all_op_latency
}


if __name__ == '__main__':
    assert len(sys.argv) > 1
    
    parser = argparse.ArgumentParser()
    parser.add_argument('func', type=str, help='specify the work to do')

    func = sys.argv[1]
    if func in function_dict.keys():
        function_dict[func](parser)
    else:
        raise ValueError(f'Function {func} not support. Supported functions: {list(function_dict.keys())}')