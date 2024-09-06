#!/bin/bash

python3 -m torch.distributed.launch \
        inference.py \
        --enc_blks 1 1 1 28  \
        --middle_blk_num 1  \
        --dec_blks 1 1  1 1 \
        --load_pre_model true \
        --experiment_name TEST6 \
        --pre_model "/app/runs/TEST_jt/NAFNet_epoch_145.pth" \
        --pre_model1 "/app/runs/TEST_jt/RDNet_epoch_145.pth" \
        --pre_model2 "/app/runs/TEST_jt/FDNet_epoch_145.pth" 
        