cd /home/sunqianpu/share_project/repo/lerobot

# ------------------nohup multi-task train------------------
# TASKS=(
#     # "agilex_build_blocks"
#     # "agilex_groceries"
#     # "agilex_fold_pants"
#     "agilex_pour_tea"
#     # "agilex_stack_basket"
#     # "agilex_pour_bowl"
#     # "agilex_scoop_bean"
#     # "agilex_seal_bag"
#     # "agilex_stir_coffee"
#     # "agilex_wipe_stains"
# )

# POLICY_TYPE=act
# # 当前可用的 GPU 数量
# NUM_GPUS=8
# GPU_ID=0

# for TASK_NAME in "${TASKS[@]}"; do
#     TIMESTAMP=$(date +%Y%m%d_%H%M%S)
#     ROOT_DIR=outputs/train/${POLICY_TYPE}
#     OUTPUT_DIR=$ROOT_DIR/${TASK_NAME}_${TIMESTAMP}
#     mkdir -p $OUTPUT_DIR
#     LOG_FILE=$OUTPUT_DIR/train.log

#     echo "Launching training for $TASK_NAME on GPU $GPU_ID"

#     CUDA_VISIBLE_DEVICES=0 python lerobot/scripts/train.py \
#       --dataset.repo_id=HuaihaiLyu/$TASK_NAME \
#       --dataset.root=/mnt/hpfs/baaiei/qianpusun/data/lerobot_data/HuaihaiLyu/$TASK_NAME \
#       --policy.type=$POLICY_TYPE \
#       --output_dir=$OUTPUT_DIR \
#       --job_name=$TASK_NAME \
#       --policy.device=cuda \
#       --steps=500000 \
#       --dataset.use_hetero=true \
#       --wandb.enable=true > $LOG_FILE 2>&1 &

#     # 轮换 GPU 编号
#     GPU_ID=$(( (GPU_ID + 1) % NUM_GPUS ))

#     # 可选：稍等几秒，避免并发冲突
#     sleep 5
# done

# # ------------------train------------------
export CUDA_VISIBLE_DEVICES=0
# agilex_build_blocks agilex_groceries agilex_fold_pants  agilex_pour_tea  agilex_stack_basket agilex_pour_bowl
POLICY_TYPE=act
TASK_NAME=/share/project/lvhuaihai/robot_data/lerobot/realman/rm_groceries
JOB_NAME=realman_baseline_groceries

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=outputs/train/${POLICY_TYPE}/${TIMESTAMP}_${JOB_NAME}
mkdir -p $OUTPUT_DIR
LOG_FILE=$OUTPUT_DIR/train.log

python lerobot/scripts/train.py \
  --dataset.repo_id=HuaihaiLyu/rm_groceries \
  --dataset.root=/share/project/lvhuaihai/robot_data/lerobot/realman/rm_groceries \
  --policy.type=$POLICY_TYPE \
  --output_dir=$OUTPUT_DIR\
  --job_name=${JOB_NAME} \
  --policy.device=cuda \
  --dataset.use_hetero=false \
  --dataset.use_relative_as=false \
  --steps=60000 \
  --wandb.enable=false 2>&1 | tee $LOG_FILE

# ------------------resume from last checkpoint------------------
# export CUDA_VISIBLE_DEVICES=5

# POLICY_TYPE=diffusion
# JOB_NAME=agilex_stack_basket_20250502_222443
# python lerobot/scripts/train.py \
#   --config_path="/mnt/hpfs/baaiei/qianpusun/ckpts/pour_tea_hetero/act/20250508_114234_agilex_pika_pour_tea/checkpoints/500000/pretrained_model/train_config.json" \
#   --steps=1000000 \
#   --resume=true \
#   --wandb.enable=false
  # 2>&1 | tee -a outputs/train/logs/$POLICY_TYPE/${JOB_NAME}_train.log
# bash /mnt/hpfs/baaiei/qianpusun/lerobot/scripts/train.sh

# find /mnt/hpfs/baaiei/qianpusun/lerobot/data/HuaihaiLyu -type d -exec cp {}/meta /mnt/hpfs/baaiei/qianpusun/ckpts/train_6_mutitask_dp_act \\;
