#!/bin/bash
# nnU-Net v2 训练：先用 plan_and_preprocess 生成 plans 与预处理数据，再逐折训练。
#
# 用法：
#   ./train.sh plan                  # 只需要跑一次
#   ./train.sh train 0 0             # fold 0, GPU 0
#   ./train.sh train 1 1
set -e

NNUNET_RAW="${nnUNet_raw:-$(pwd)/nnUNet_raw}"
export nnUNet_raw="$NNUNET_RAW"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$NNUNET_RAW/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$NNUNET_RAW/nnUNetresults}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

DATASET="${NME_TASK:-Dataset001}"
CONFIG="${NME_CONFIG:-3d_fullres}"

case "$1" in
  plan)
    echo "[$(date '+%F %T')] plan_and_preprocess $DATASET"
    nnUNetv2_plan_and_preprocess -d "${DATASET#Dataset}" -c "$CONFIG" --verify_dataset_integrity
    ;;
  train)
    FOLD="${2:?需要 fold}"; GPU="${3:?需要 GPU id}"
    export CUDA_VISIBLE_DEVICES="$GPU"
    echo "[$(date '+%F %T')] START fold=$FOLD gpu=$GPU"
    exec nnUNetv2_train "$DATASET" "$CONFIG" "$FOLD"
    ;;
  *)
    echo "用法: $0 {plan|train <fold> <gpu>}"; exit 1
    ;;
esac
