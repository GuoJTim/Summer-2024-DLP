python3 -m torch.distributed.launch  --nproc_per_node=4  \
        training.py \
        --enc_blks 1 1 1 28  \
        --middle_blk_num 1  \
        --dec_blks 1 1  1 1 \
        --BATCH_SIZE 3 \
        --experiment_name TEST_jt_2 \
        --learning_rate 0.0006 \
        --gamma_rd 0.0001 \
        --gamma_rr 0.02 \
        --gamma_fd 0.0005 \
        --RDNet_Loss_func char \
        --FDNet_Loss_func char \
        --RRNet_Other_Loss ssim \
        --EPOCH 1000