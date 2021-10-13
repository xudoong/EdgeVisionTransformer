import sys
import subprocess
import random
import string
import os

template = \
"""
description: Are16heads DeiT exp ({job_name})

target:
  service: amlk8s
  # run "pt target list amlk8s" to list the names of available AMLK8s targets
  name: itpscusv100cl # TODO
  vc: resrchvc

environment:
  image: taoky/pytorch-1.8.1-cuda10.2-cudnn7-devel-enhanced:latest
  # sh does not have "source", so use ". ./xxx.sh" here.
  setup:
  - . ./itp/setup.sh

code:
  # upload the code
  local_dir: $CONFIG_DIR/../../

storage:
  teamdrive:
    storage_account_name: hexnas
    container_name: teamdrive
    mount_dir: /mnt/data
    local_dir: $CONFIG_DIR/../../../faketeamdrive/


jobs:
{jobs}
"""

job_template = \
"""- name: {job_name}
  sku: G{gpu_cnt}
  command:
  - ./itp/run_itp.sh {function}
"""

func_to_job_name_dict = {
  'iterative_pruning_base': 'D1009_are16heads_iterative_pruning_deit_base',
  'finetune_many_base': 'D1013_are16heads_finetune_pruned_deit_base'
}
def main(mode):
  function = sys.argv[2]
  assert function in func_to_job_name_dict.keys()
  
  job_name = func_to_job_name_dict[function]  # !! Edit this
  jobs = ""
  jobs += job_template.format(
      job_name=job_name, gpu_cnt=4, function=function
  )
  description = f"EdgeDL Are16heads exp ({job_name})"

  # ======================================================================================================
  # Don't need to modify following code
  result = template.format(
      job_name=job_name,
      jobs=jobs,
  )
  print(result)

  tmp_name = ''.join(random.choices(string.ascii_lowercase, k=6)) + job_name
  tmp_name = os.path.join("./.tmp", tmp_name)
  with open(tmp_name, "w") as fout:
    fout.write(result)
  if mode == 0:
    subprocess.run(["amlt", "run", "-t", "local", "--use-sudo", tmp_name, "--devices", "all"])
    input()
  elif mode == 1:
    subprocess.run(["amlt", "run", "-d", description, tmp_name, job_name])


if __name__ == "__main__":
  # example: python xx.py submit tiny 50
  # tiny (sys.argv[2]) is deit_type
  # 50 (sys.argv[2]) is sparsity
  if len(sys.argv) == 2 and sys.argv[1] in ('--help', '-h'):
    print('Example cmd: python <this_file> submit iterative_pruning_base')
    exit()
  mode = 2
  if len(sys.argv) == 3 and sys.argv[1] == 'submit':
    print("Submit (pt run)")
    mode = 1
  elif len(sys.argv) == 3 and sys.argv[1] == 'debug':
    print("Debug dry run (pt run -t local)")
    mode = 0
  else:
    print("Print only")

  main(mode)
