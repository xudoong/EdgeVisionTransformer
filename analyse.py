import argparse
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

def _find_begin_line(rows):
    schema = {}
    for begin_line in range(len(rows)):
        row = rows[begin_line]
        if len(row) == 1 and 'Operator-wise Profiling Info for Regular Benchmark Run' in row[0]:
            schema_row = rows[begin_line + 2]
            schema = {schema_row[i].strip(): i for i in range(len(schema_row))}
            begin_line += 3
            break
    return begin_line, schema

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
    begin_line, schema = _find_begin_line(rows)
    print(f'Schema: {schema}')

    result_table = {}
    for i in range(begin_line, len(rows)):
        row = rows[i]
        if len(row) < len(schema):
            break
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
    begin_line, schema = _find_begin_line(rows)
    print(f'Schema: {schema}')

    gelu_latency = 0
    gelu_percent = 0
    ln_latency = 0
    ln_percent = 0

    hit_gelu = 0
    hit_ln = 0
    for i in range(begin_line, len(rows)):
        row = rows[i]
        if len(row) < len(schema):
            break
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

    
    print('hit_gelu{} hit_ln{} gelu_latency{:.2f} gelu_percent{:.2f} ln_latency{:.2f} ln_percent{:.2f}'.format(
        hit_gelu, hit_ln, gelu_latency, gelu_percent, ln_latency, ln_percent))


def analyse_attn_ffn(parser: argparse.ArgumentParser):
    import csv
    parser.add_argument('--file', type=str, required=True, help='csv profile result file')
    parser.add_argument('--type', choices=['deit', 'swin', 't2t_vit'], required=True, help='the transformer model type')
    args = parser.parse_args()

    rows = _read_rows(args.file)
    begin_line, schema = _find_begin_line(rows)
    print(f'Schema: {schema}')

    attn_percent = 0
    attn_latency = 0
    ffn_percent = 0
    ffn_latency = 0
    pre_post_processing_percent = 0
    pre_post_processing_latency = 0

    for i in range(begin_line, len(rows)):
        row = rows[i]
        if len(row) < len(schema):
            break
        node_type = row[schema['node type']]

        if args.type == 'swin':
            if 'window_attention' in rows[i][schema['name']]:
                attn_latency += float(rows[i][schema['avg_ms']])
                attn_percent += float(rows[i][schema['%']][:-1])
            if 'mlp' in rows[i][schema['name']]:
                ffn_latency += float(rows[i][schema['avg_ms']])
                ffn_percent += float(rows[i][schema['%']][:-1])
            if 'sequential_4' not in rows[i][schema['name']]:
                pre_post_processing_latency += float(rows[i][schema['avg_ms']])
                pre_post_processing_percent += float(rows[i][schema['%']][:-1])

        if args.type == 'deit':
            if 'attention' in rows[i][schema['name']]:
                attn_latency += float(rows[i][schema['avg_ms']])
                attn_percent += float(rows[i][schema['%']][:-1])
            if 'feed_forward' in rows[i][schema['name']]:
                ffn_latency += float(rows[i][schema['avg_ms']])
                ffn_percent += float(rows[i][schema['%']][:-1])
            if 'transformer_encoder_block' not in rows[i][schema['name']]:
                pre_post_processing_latency += float(rows[i][schema['avg_ms']])
                pre_post_processing_percent += float(rows[i][schema['%']][:-1])

        if args.type == 't2t_vit':
            if 'transformer_encoder_block' in rows[i][schema['name']] and 'attention' in rows[i][schema['name']]:
                attn_latency += float(rows[i][schema['avg_ms']])
                attn_percent += float(rows[i][schema['%']][:-1])
            if 'transformer_encoder_block' in rows[i][schema['name']] and 'feed_forward' in rows[i][schema['name']]:
                ffn_latency += float(rows[i][schema['avg_ms']])
                ffn_percent += float(rows[i][schema['%']][:-1])
            if 'transformer_encoder_block' not in rows[i][schema['name']]:
                pre_post_processing_latency += float(rows[i][schema['avg_ms']])
                pre_post_processing_percent += float(rows[i][schema['%']][:-1])

    print(f'{args.type} | attn (percent, latency) = ({attn_percent:.2f}, {attn_latency:.2f}) | ' + 
          f'ffn (percent, latency) = ({ffn_percent:.2f}, {ffn_latency:.2f}) | ' +
          f'pre & post-processing (percent, latency) = ({pre_post_processing_percent:.2f}, {pre_post_processing_latency:.2f})')


function_dict = {
    'analyse_op': analyse_op,
    'analyse_gelu_ln': analyse_gelu_ln,
    'analyse_attn_ffn': analyse_attn_ffn,
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