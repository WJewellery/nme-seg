"""路径与实验标识，全部通过环境变量配置。

nnU-Net 自身读取 nnUNet_raw / nnUNet_preprocessed / nnUNet_results 三个环境变量。
本文件只做统一入口，避免脚本里散落硬编码路径。

默认约定（可用环境变量覆盖）：

    ./nnUNet_raw/                     <- nnUNet_raw
    ├── Dataset001/                   <- 原始数据（make_dataset.py 的输出）
    ├── nnUNet_preprocessed/          <- plan_and_preprocess 的输出
    └── nnUNetresults/                <- 训练输出

    ./data/                           <- 原始 NIfTI（含 imagesTr/ 与 labelsTr/）
"""
import os

TASK = os.environ.get("NME_TASK", "Dataset001")
TRAINER = os.environ.get("NME_TRAINER", "nnUNetTrainer")
PLANS = os.environ.get("NME_PLANS", "nnUNetPlans")
CONFIG = os.environ.get("NME_CONFIG", "3d_fullres")

# nnU-Net 标准环境变量，未设置时按上述目录约定给出
NNUNET_RAW = os.environ.setdefault("nnUNet_raw", os.path.join(os.getcwd(), "nnUNet_raw"))
NNUNET_PRE = os.environ.setdefault("nnUNet_preprocessed",
                                   os.path.join(NNUNET_RAW, "nnUNet_preprocessed"))
NNUNET_RES = os.environ.setdefault("nnUNet_results", os.path.join(NNUNET_RAW, "nnUNetresults"))

DATA_DIR = os.path.join(NNUNET_RAW, TASK)
MODEL_DIR = os.path.join(NNUNET_RES, TASK, f"{TRAINER}__{PLANS}__{CONFIG}")

# 原始影像与掩膜的源目录（make_dataset.py 的输入）
SRC_ROOT = os.environ.get("NME_SEG_SRC", os.path.join(os.getcwd(), "data"))
