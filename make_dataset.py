#!/usr/bin/env python
"""原始 NIfTI → nnU-Net v2 Dataset00X 目录结构。

输入约定：
    SRC/imagesTr/<case>.nii.gz     原始 T1+C 首期影像
    SRC/labelsTr/<case>.nii.gz     专家标注掩膜（0 背景 / 1 病灶）

输出（nnU-Net v2 标准布局）：
    OUT/imagesTr/<case>_0000.nii.gz
    OUT/labelsTr/<case>.nii.gz
    OUT/dataset.json

用法：
    python make_dataset.py --src ./data --out $nnUNet_raw/Dataset001
"""
import argparse, json, os, shutil, sys
import numpy as np
import nibabel as nib


def check_pair(img_path, seg_path):
    """影像与掩膜必须在同一网格上，否则 nnU-Net 预处理会失败。"""
    img = nib.load(img_path)
    seg = nib.load(seg_path)
    if img.shape != seg.shape:
        return f"形状不一致 影像{img.shape} vs 掩膜{seg.shape}"
    if not np.allclose(img.header.get_zooms()[:3], seg.header.get_zooms()[:3], atol=1e-3):
        return f"体素间距不一致 {img.header.get_zooms()[:3]} vs {seg.header.get_zooms()[:3]}"
    if not np.allclose(img.affine, seg.affine, atol=1e-3):
        return "affine 不一致"
    lab = np.unique(np.asanyarray(seg.dataobj))
    bad = [v for v in lab if v not in (0, 1)]
    if bad:
        return f"掩膜取值非二值: {sorted(set(bad))[:5]}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="含 imagesTr/ 与 labelsTr/ 的源目录")
    ap.add_argument("--out", required=True, help="输出 Dataset00X 目录")
    ap.add_argument("--channel_name", default="CT",
                    help="nnU-Net 通道名。默认 CT 是历史遗留命名，会触发 CTNormalization "
                         "（0.5/99.5 百分位裁剪后 z-score），与本项目报告结果一致；"
                         "改成 T1C 会切换为 ZScoreNormalization，需重新训练。")
    ap.add_argument("--label_name", default="tumor")
    ap.add_argument("--copy", action="store_true", help="复制而非建软链接")
    ap.add_argument("--allow_mismatch", action="store_true", help="跳过几何校验（不推荐）")
    args = ap.parse_args()

    img_dir = os.path.join(args.src, "imagesTr")
    seg_dir = os.path.join(args.src, "labelsTr")
    if not os.path.isdir(img_dir):
        sys.exit(f"找不到 {img_dir}")

    out_img = os.path.join(args.out, "imagesTr")
    out_seg = os.path.join(args.out, "labelsTr")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_seg, exist_ok=True)

    cases = sorted(f[:-7] for f in os.listdir(img_dir) if f.endswith(".nii.gz"))
    if not cases:
        sys.exit(f"{img_dir} 里没有 .nii.gz")

    bad, kept = [], []
    for i, c in enumerate(cases, 1):
        img_p = os.path.join(img_dir, c + ".nii.gz")
        seg_p = os.path.join(seg_dir, c + ".nii.gz")
        if not os.path.exists(seg_p):
            bad.append((c, "缺掩膜")); continue
        if not args.allow_mismatch:
            msg = check_pair(img_p, seg_p)
            if msg:
                bad.append((c, msg)); continue
        dst_i = os.path.join(out_img, c + "_0000.nii.gz")
        dst_s = os.path.join(out_seg, c + ".nii.gz")
        for src_p, dst_p in ((img_p, dst_i), (seg_p, dst_s)):
            if os.path.exists(dst_p):
                os.remove(dst_p)
            if args.copy:
                shutil.copy2(src_p, dst_p)
            else:
                os.symlink(os.path.abspath(src_p), dst_p)
        kept.append(c)
        if i % 50 == 0 or i == len(cases):
            print(f"  [{i}/{len(cases)}] {c}", flush=True)

    meta = {
        "channel_names": {"0": args.channel_name},
        "labels": {"background": 0, args.label_name: 1},
        "numTraining": len(kept),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    with open(os.path.join(args.out, "dataset.json"), "w") as f:
        json.dump(meta, f, indent=4)

    print(f"\n完成：{len(kept)} 例 → {args.out}")
    print(f"dataset.json: {json.dumps(meta, ensure_ascii=False)}")
    if bad:
        print(f"\n跳过 {len(bad)} 例：")
        for c, why in bad:
            print(f"  {c}: {why}")


if __name__ == "__main__":
    main()
