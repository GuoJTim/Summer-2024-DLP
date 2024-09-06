#!/bin/bash

python3 -m torch.distributed.launch --nproc_per_node=4 \
        training.py \
        --enc_blks 1 1 1 28  \
        --middle_blk_num 1  \
        --dec_blks 1 1  1 1 \
        --BATCH_SIZE 3 \
        --START_EPOCH 474 \
        --EPOCH 1000 \
        --load_pre_model true \
        --experiment_name TEST_FINETUNE \
        --pre_model "/app/runs/TEST/NAFNet_epoch_475.pth" \
        --pre_model1 "/app/runs/TEST/RDNet_epoch_475.pth" \
        --pre_model2 "/app/runs/TEST/FDNet_epoch_475.pth" 
        