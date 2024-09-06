for mask_func in linear cosine square
do
    iter=0
    # 不斷嘗試訪問資料夾，直到該資料夾不存在
    while true
    do
        folder="../loss_all_cross_entropy_test_results/${mask_func}_it_${iter}"
        
        # 如果資料夾存在，則處理該資料夾
        if [ -d "$folder" ]; then
            echo "Processing folder: $folder"
            python3 fid_score_gpu.py \
                --predicted-path $folder \
                --device cuda \
                --mask_fun $mask_func \
                --iter $iter
            ((iter++))  # 增加 iter
        else
            # 如果資料夾不存在，則退出迴圈
            break
        fi
    done
done