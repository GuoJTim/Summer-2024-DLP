#!/bin/bash

python3 -m torch.distributed.launch \
        inference.py \
        --enc_blks 1 1 1 28  \
        --middle_blk_num 1  \
        --dec_blks 1 1  1 1 \
        --load_pre_model true \
        --experiment_name TEST7 \
        --pre_model "/app/runs/TEST_jt_2/NAFNet_epoch_125.pth" \
        --pre_model1 "/app/runs/TEST_jt_2/RDNet_epoch_125.pth" \
        --pre_model2 "/app/runs/TEST_jt_2/FDNet_epoch_125.pth" 
        