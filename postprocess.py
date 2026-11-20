#!/usr/bin/env python3
"""nnU-Net 乳腺 MR 肿块分割 —— 预测掩膜后处理

两步，对应 Methods 里的描述：
  1) 3D 26-邻接连通域分析，去掉孤立小连通域（默认 <50 mm³）
  2) 最小病灶体积阈值（默认 250 mm³）：总体积不足则判为无病灶，输出空掩膜

用法：
  python postprocess.py --in_dir PRED_DIR --out_dir OUT_DIR
  python postprocess.py --in_dir PRED_DIR --out_dir OUT_DIR --min_component_mm3 50 --min_lesion_mm3 250
"""
import argparse, csv, os, sys
import numpy as np
import nibabel as nib
from skimage.measure import label


def postprocess_mask(mask, spacing, min_component_mm3=50.0, min_lesion_mm3=250.0):
    """返回 (处理后掩膜 uint8, 统计信息 dict)。mask: 0/1 或概率。"""
    vox = float(np.prod(spacing))
    binary = mask > 0.5
    vox_before = int(binary.sum())

    # --- 步骤 1：连通域分析，去掉孤立小连通域 ---
    lab, n = label(binary, connectivity=3, return_num=True)
    if n == 0:
        stats = dict(n_components=0, n_kept=0, vox_before=0, vox_after_cc=0,
                     volume_mm3_before=0.0, volume_mm3_after_cc=0.0,
                     volume_mm3_final=0.0, is_positive=0, reason="empty")
        return np.zeros(binary.shape, dtype=np.uint8), stats
    sizes = np.bincount(lab.ravel())
    keep = [i for i in range(1, n + 1) if sizes[i] * vox >= min_component_mm3]
    cc = np.isin(lab, keep) if keep else np.zeros(binary.shape, dtype=bool)
    vox_cc = int(cc.sum())

    # --- 步骤 2：最小病灶体积阈值 ---
    vol_cc = vox_cc * vox
    if vol_cc < min_lesion_mm3:
        out = np.zeros(binary.shape, dtype=np.uint8)
        reason = "below_min_lesion_volume"
    else:
        out = cc.astype(np.uint8)
        reason = "kept"

    stats = dict(
        n_components=int(n), n_kept=len(keep),
        vox_before=vox_before, vox_after_cc=vox_cc, vox_final=int(out.sum()),
        volume_mm3_before=round(vox_before * vox, 2),
        volume_mm3_after_cc=round(vol_cc, 2),
        volume_mm3_final=round(float(out.sum()) * vox, 2),
        is_positive=int(out.sum() > 0), reason=reason,
    )
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_component_mm3", type=float, default=50.0,
                    help="小于该体积的连通域视为孤立小区域并删除（默认 50 mm³）")
    ap.add_argument("--min_lesion_mm3", type=float, default=250.0,
                    help="总体积小于该值时判为无病灶（默认 250 mm³）")
    ap.add_argument("--csv", default=None, help="逐病例统计输出路径")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(args.in_dir) if f.endswith(".nii.gz"))
    if not names:
        sys.exit(f"错误：{args.in_dir} 里没有 .nii.gz")

    rows = []
    for i, name in enumerate(names, 1):
        img = nib.load(os.path.join(args.in_dir, name))
        arr = np.asanyarray(img.dataobj)
        spacing = img.header.get_zooms()[:3]
        out, st = postprocess_mask(arr, spacing, args.min_component_mm3, args.min_lesion_mm3)

        new = nib.Nifti1Image(out, img.affine)
        new.header.set_data_dtype(np.uint16)
        nib.save(new, os.path.join(args.out_dir, name))

        row = dict(case=name, **st)
        rows.append(row)
        print(f"[{i}/{len(names)}] {name:24s} "
              f"连通域 {st['n_components']:>2d}→保留 {st['n_kept']:>2d}  "
              f"体积 {st['volume_mm3_before']:>10.1f}→{st['volume_mm3_final']:>10.1f} mm³  {st['reason']}",
              flush=True)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    pos = sum(r["is_positive"] for r in rows)
    print(f"\n完成：{len(rows)} 例，判为阳性 {pos} 例，阴性 {len(rows)-pos} 例")
    print(f"连通域删除总计: {sum(r['n_components']-r['n_kept'] for r in rows)} 个")
    print(f"体积被 250 mm³ 阈值清零: {sum(1 for r in rows if r['reason']=='below_min_lesion_volume')} 例")


if __name__ == "__main__":
    main()
