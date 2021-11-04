from supernet import SwiftBERT
import random
import torch
from glob import glob
from pathlib import Path
import json
from transformers import BertConfig
from supernet import SwiftBERTOutput
import argparse
baseconfig={
  "_name_or_path": "google/bert_uncased_L-4_H-256_A-4",
  "architectures": [
    "SwiftBERT"
  ],
  "attention_probs_dropout_prob": 0.1,
  "gradient_checkpointing": False,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 256,
  "initializer_range": 0.02,
  "intermediate_size": 1024,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 4,
  "num_hidden_layers": 4,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.7.0",
  "type_vocab_size": 2,
  "use_cache": True,
  "vocab_size": 30522
}
def gen_testconfigs(sample_num):
    heads_nums=[0.25,0.5,0.75,1]
    intermediate_sizes=[a/100.0 for a in list(range(1,101))]
    
    #sample_num=100

    for si in range(sample_num):
        curconfig=baseconfig

        ## generate a test file
        curconfig['layers']={}
        tag=""
        for layer in range(4):
            head_num=random.sample(heads_nums,1)[0]
            im_size=random.sample(intermediate_sizes,1)[0]
            print(layer,head_num,im_size)
            curconfig['layers'][layer]={}
            curconfig['layers'][layer]['heads']=int(head_num*4)
            curconfig['layers'][layer]['intermediate_size']=int(im_size*1024) 
            tag+='h_'+str(head_num)+'_d_'+str(im_size)+'-'

        tag=tag[0:-1]
        fw=open('latency_data/'+tag+'.json','w')
        fw.write(json.dumps(curconfig,indent=4))
def gen_original():
    curconfig=baseconfig

        ## generate a test file
    curconfig['layers']={}
    tag=""
    for layer in range(4):
            head_num=4
            im_size=1024
            print(layer,head_num,im_size)
            curconfig['layers'][layer]={}
            curconfig['layers'][layer]['heads']=head_num
            curconfig['layers'][layer]['intermediate_size']=im_size 
            tag+='h_'+str(head_num)+'_d_'+str(im_size)+'-'

    tag='config'
    fw=open('latency_data/'+tag+'.json','w')
    fw.write(json.dumps(curconfig,indent=4))

def gen_uniform():
    for h in range(1,5):
        for j in range(1,101):

            curconfig=baseconfig

            ## generate a test file
            curconfig['layers']={}
            tag=""
            for layer in range(4):
                head_num=h
                im_size=1024
                print(layer,head_num,im_size)
                curconfig['layers'][layer]={}
                curconfig['layers'][layer]['heads']=head_num
                curconfig['layers'][layer]['intermediate_size']=int(im_size*(j/100.0)) 
                tag+='h_'+str(head_num/4.0)+'_d_'+str(j/100.0)+'-'

            print(tag)
            fw=open('latency_data/'+tag[0:-1]+'.json','w')
            fw.write(json.dumps(curconfig,indent=4))



#gen_testconfigs()


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=Path, default='latency_data')
parser.add_argument("--nn_pruning", action='store_true')
parser.add_argument("--no_opt", action='store_true')
parser.add_argument("--force_opt", action='store_true')
parser.add_argument("--max_ad_length", type=int, default=38)
parser.add_argument("--output_name", type=str, default="output")
parser.add_argument("--opset_version", type=int, default=13)

args = parser.parse_args()
assert not (args.no_opt and args.force_opt), "no_opt and force_opt cannot be set together."
# python src/onnx_export.py --model_dir ./results/dummy_mini/final/
def gen_onnx(config):
    myconfig=BertConfig.from_pretrained(config)
    print(myconfig)
    model = SwiftBERTOutput(myconfig)
    print(model)
    bert_config = model.config

    max_ad_length = args.max_ad_length

    print("==== export ====")
    output_name = config.replace(".json","")

    torch.onnx.export(
    model,
    (torch.tensor([1] * (max_ad_length)).view(-1, max_ad_length),
        torch.tensor([1] * (max_ad_length)).view(-1, max_ad_length),
        torch.tensor([1] * (max_ad_length)).view(-1, max_ad_length)),
     f'{output_name}.onnx', 
    input_names=['input_ids', 'attention_mask', 'token_type_ids'],
    output_names=['score'],
    verbose=False,
    export_params=True,
    opset_version=args.opset_version,
    do_constant_folding=True
    )

'''
gen_original()
gen_onnx('latency_data/config.json')
'''

#gen_testconfigs(2000)
#gen_original()
#gen_uniform()
filenames=glob('latency_data/**.json')
for filename in filenames:
    gen_onnx(filename)
