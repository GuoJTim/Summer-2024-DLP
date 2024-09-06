#!/bin/bash

python3 -m torch.distributed.launch \
        inference_no_FDNet.py \
        --enc_blks 1 1 1 28  \
        --middle_blk_num 1  \
        --dec_blks 1 1  1 1 \
        --load_pre_model true \
        --experiment_name test_org \
        --pre_model "/app/src/ReflectionRemoval_basic/ckpt/RR.pth" \
        --pre_model1 "/app/src/ReflectionRemoval_basic/ckpt/RD.pth"
        