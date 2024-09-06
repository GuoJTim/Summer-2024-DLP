#!/bin/bash

python3 -m torch.distributed.launch --nproc_per_node=4 \
        training.py \
        --enc_blks 1 1 1 28  \
        --middle_blk_num 1  \
        --dec_blks 1 1  1 1 \
        --BATCH_SIZE 3 \
        --EPOCH 1000 \
        --experiment_name TEST 