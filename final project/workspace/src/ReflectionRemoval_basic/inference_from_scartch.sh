#!/bin/bash

python3 -m torch.distributed.launch \
        inference.py \
        --enc_blks 1 1 1 28  \
        --middle_blk_num 1  \
        --dec_blks 1 1  1 1 \
        --load_pre_model true \
        --experiment_name TEST5 \
        --pre_model "/app/runs/TEST_FROMSCRATCH/NAFNet_epoch_165.pth" \
        --pre_model1 "/app/runs/TEST_FROMSCRATCH/RDNet_epoch_165.pth" \
        --pre_model2 "/app/runs/TEST_FROMSCRATCH/FDNet_epoch_165.pth" 
        