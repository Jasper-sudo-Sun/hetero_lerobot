#!/bin/bash
cd /share/project/lvhuaihai/lvhuaihai/hetero_lerobot

POLICY_TYPE=act

# 所有要训练的数据集名称
DATASETS=(
    # "pika_pour_banana"
    "pika_pour_banana_delta"
    # "pika_pour_banana_withoutcentric"
    # "pika_pour_banana_withoutcentric_delta"
)

# GPU起始编号和可用总数
START_GPU=0
NUM_GPUS=4
GPU_ID=$START_GPU

for DATASET_NAME in "${DATASETS[@]}"; do
    DATASET_ROOT="/share/project/lvhuaihai/robot_data/lerobot/HuaihaiLyu/${DATASET_NAME}"
    MACHINE_NAME=$(echo $DATASET_NAME | cut -d'_' -f1)
    JOB_NAME="${DATASET_NAME}_pika"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="outputs/train/${MACHINE_NAME}/${POLICY_TYPE}/${JOB_NAME}_${TIMESTAMP}"
    LOG_FILE="${OUTPUT_DIR}/train.log"

    mkdir -p $OUTPUT_DIR

    echo "Launching training for $DATASET_NAME on GPU $GPU_ID"

    CUDA_VISIBLE_DEVICES=$GPU_ID nohup python lerobot/scripts/train.py \
      --dataset.repo_id=HuaihaiLyu/${DATASET_NAME} \
      --dataset.root=${DATASET_ROOT} \
      --policy.type=${POLICY_TYPE} \
      --output_dir=${OUTPUT_DIR} \
      --job_name=${JOB_NAME} \
      --policy.device=cuda \
      --dataset.use_hetero=false \
      --dataset.use_relative_as=false \
      --steps=500000 \
      --wandb.enable=true > ${LOG_FILE} 2>&1 &

    # GPU编号轮换
    GPU_ID=$(( (GPU_ID + 1) % NUM_GPUS ))

    # 稍作延迟，避免冲突
    sleep 3
done
