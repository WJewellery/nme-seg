#!/usr/bin/env python
"""nnU-Net v2 推理：从训练好的模型目录对一批影像做预测。

输入目录要求 nnU-Net 的命名规则：<case>_0000.nii.gz（单通道）。
可用 --rename 把不带通道后缀的文件自动补成 _0000。

用法：
    python predict.py --input ./demo/in --output ./out/raw --folds 3 \
        --model_dir $nnUNet_results/Dataset001/nnUNetTrainer__nnUNetPlans__3d_fullres

    # 5 折集成
    python predict.py --input ./in --output ./out/raw --folds all
"""
import argparse, os, shutil, sys, time

# 先加载 config 以设置 nnUNet_* 环境变量，避免 nnU-Net 启动时反复告警
try:
    import config  # noqa: F401
except Exception:
    pass

import torch


def normalize_input(in_dir, work_dir):
    """把任意命名的 .nii.gz 复制成 nnU-Net 要求的 <case>_0000.nii.gz。"""
    os.makedirs(work_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(in_dir) if f.endswith(".nii.gz"))
    if not names:
        sys.exit(f"{in_dir} 里没有 .nii.gz")
    out = []
    for n in names:
        if n.endswith("_0000.nii.gz"):
            stem = n
        else:
            stem = n[:-7] + "_0000.nii.gz"
        shutil.copy2(os.path.join(in_dir, n), os.path.join(work_dir, stem))
        out.append(stem)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="待预测影像目录")
    ap.add_argument("--output", required=True, help="预测掩膜输出目录")
    ap.add_argument("--model_dir", default=os.environ.get("NME_MODEL_DIR"),
                    help="训练输出目录（含 fold_*/ 与 plans.json）")
    ap.add_argument("--folds", default="3", help="'3' 或 'all' 或 '0,1,2,3,4'")
    ap.add_argument("--checkpoint", default="checkpoint_best.pth")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tile_step_size", type=float, default=0.5)
    ap.add_argument("--no_mirroring", action="store_true", help="关闭测试时镜像增强")
    ap.add_argument("--save_probabilities", action="store_true")
    ap.add_argument("--pre_workers", type=int, default=2)
    ap.add_argument("--export_workers", type=int, default=2)
    args = ap.parse_args()

    if not args.model_dir:
        sys.exit("需要 --model_dir 或环境变量 NME_MODEL_DIR")

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    if args.folds.strip() == "all":
        folds = tuple(range(5))
    else:
        folds = tuple(int(x) for x in args.folds.split(",") if x.strip())

    work = os.path.join(args.output, "_nnunet_input")
    stems = normalize_input(args.input, work)
    os.makedirs(args.output, exist_ok=True)

    t0 = time.time()
    p = nnUNetPredictor(tile_step_size=args.tile_step_size,
                        use_gaussian=True,
                        use_mirroring=not args.no_mirroring,
                        device=torch.device(args.device),
                        verbose=False)
    p.initialize_from_trained_model_folder(args.model_dir, use_folds=folds,
                                           checkpoint_name=args.checkpoint)
    p.predict_from_files(work, args.output, save_probabilities=args.save_probabilities,
                         num_processes_preprocessing=args.pre_workers,
                         num_processes_segmentation_export=args.export_workers)

    # 还原成不带通道后缀的文件名
    for s in stems:
        src = os.path.join(args.output, s)
        dst = os.path.join(args.output, s.replace("_0000.nii.gz", ".nii.gz"))
        if os.path.exists(src) and src != dst:
            shutil.move(src, dst)

    print(f"完成：{len(stems)} 例，{time.time()-t0:.1f}s → {args.output}")
    print("提示：原始预测掩膜未经后处理，下一步跑 postprocess.py")


if __name__ == "__main__":
    main()
