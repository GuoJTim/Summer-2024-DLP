for mask_func in linear cosine square
do
    python3 inpainting.py \
        --test_maskedimage_path ./lab5_dataset/lab5_dataset/masked_image \
        --test_mask_path ./lab5_dataset/lab5_dataset/mask64 \
        --load_transformer_ckpt_path ./transformer_checkpoints/transformer_epoch_50.pt \
        --mask_func $mask_func \
        --total_iter 19 \
        --sweet_spot 42
done
