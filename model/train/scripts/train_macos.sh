DATASET_PATH=$1
MODEL_PATH=$2

GLOBAL_BATCH_SIZE=8
MICRO_BATCH_SIZE=1
GPUS_PER_NODE=1
GRAD_ACC=$((${GLOBAL_BATCH_SIZE} / (${GPUS_PER_NODE} * ${MICRO_BATCH_SIZE}) ))

NUM_EPOCHS=2
OUTPUT_DIR="model/train/output_dir"

python model/train/train_macos.py \
    --train_path $DATASET_PATH \
    --model_name_or_path $MODEL_PATH \
    --per_device_train_batch_size $MICRO_BATCH_SIZE \
    --max_len 2048 \
    --max_src_len 1536 \
    --num_train_epochs $NUM_EPOCHS \
    --gradient_accumulation_steps $GRAD_ACC \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.1 \
    --mode llama \
    --train_type lora \
    --lora_dim 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_module_name "q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj" \
    --seed 1234 \
    --save_model_step 2000 \
    --show_loss_step 50 \
    --output_dir $OUTPUT_DIR
