env CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python3 -u finetune.py \
    --data_dir ./data \
    --train_name train \
    --output_dir=./model \
    --save_top_k 60 \
    --train_batch_size=12 \
    --eval_batch_size=12 \
    --num_train_epochs 10 \
    --model_name_or_path ./LongLM-small \
    --learning_rate=3e-5 \
    --fp16 \
    --gpus 1 \
    --do_train \
    --n_val 100 \
    --val_check_interval 1.0 \
    --gradient_accumulation_steps 1 \
    --overwrite_output_dir