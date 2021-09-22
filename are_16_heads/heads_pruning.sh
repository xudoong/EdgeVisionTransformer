#!/usr/bin/env bash

TASK=$1
OPTIONS="${@:2}"

here="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $here/prepare_task.sh $TASK

echo $base_acc
prune_options="--do_prune --eval_pruned --prune_percent `seq 5 5 100` $OPTIONS"
run_eval "$prune_options"

# prune cmd: bash experiments/BERT/heads_pruning.sh MNLI --normalize_pruning_by_layer