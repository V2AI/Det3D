#!/bin/bash
CONFIG=$1
WORK_DIR=$2
CHECKPOINT=$3

# Test
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    ./tools/dist_test.py \
    $CONFIG \
    --work_dir=$WORK_DIR \
    --checkpoint=$CHECKPOINT \


