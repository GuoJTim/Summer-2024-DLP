python3 -m torch.distributed.launch \
        training.py \
        --local_rank 0 \
        --enc_blks 1 1 1 28  \
        --middle_blk_num 1  \
        --dec_blks 1 1  1 1 \
        --BATCH_SIZE 8 \
        --experiment_name TEST_royyang \
        --RRNet_Other_Loss ssim \
        --EPOCH 1000