#!/bin/bash

# 激活环境
source activate lerobot || conda activate lerobot

cd /mnt/hpfs/baaiei/qianpusun/lerobot

# 定义你的任务列表
TASKS=(
    # "agilex_build_blocks"
    # "agilex_groceries"
    # "agilex_fold_pants"
    # "agilex_pour_tea"
    # "agilex_stack_basket"
    # "agilex_pour_bowl"
    "agilex_scoop_bean"
    # "agilex_seal_bag"
    # "agilex_stir_coffee"
    # "agilex_wipe_stains"
)

POLICY_TYPE=act

# 当前可用的 GPU 数量
NUM_GPUS=8
GPU_ID=0

for TASK_NAME in "${TASKS[@]}"; do
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    ROOT_DIR=outputs/train2/${POLICY_TYPE}
    OUTPUT_DIR=$ROOT_DIR/${TASK_NAME}_${TIMESTAMP}
    mkdir -p $OUTPUT_DIR
    LOG_FILE=$OUTPUT_DIR/train.log

    echo "Launching training for $TASK_NAME on GPU $GPU_ID"

    CUDA_VISIBLE_DEVICES=$GPU_ID nohup python lerobot/scripts/train.py \
      --dataset.repo_id=HuaihaiLyu/$TASK_NAME \
      --dataset.root=/mnt/hpfs/baaiei/qianpusun/data/lerobot_data/HuaihaiLyu/$TASK_NAME \
      --policy.type=$POLICY_TYPE \
      --output_dir=$OUTPUT_DIR \
      --job_name=$TASK_NAME \
      --policy.device=cuda \
      --steps=500000 \
      --wandb.enable=true > $LOG_FILE 2>&1 &

    # 轮换 GPU 编号
    GPU_ID=$(( (GPU_ID + 1) % NUM_GPUS ))

    # 可选：稍等几秒，避免并发冲突
    sleep 5
done
