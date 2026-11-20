#!/usr/bin/env python
"""生成合成体模（不含任何病人数据），用于在没有真实数据时跑通流程。

产物：
    demo/data/imagesTr/phantom_0001.nii.gz
    demo/data/labelsTr/phantom_0001.nii.gz

用法：
    python demo/make_phantom.py --out ./demo/data
"""
import argparse, os
import numpy as np
import nibabel as nib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./demo/data")
    ap.add_argument("--shape", default="96,128,128", help="z,y,x 体素")
    ap.add_argument("--spacing", default="1.0,0.76,0.76", help="z,y,x mm")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    shape = tuple(int(v) for v in args.shape.split(","))
    spacing = tuple(float(v) for v in args.spacing.split(","))
    rng = np.random.RandomState(args.seed)

    z, y, x = [np.arange(n) * s - n * s / 2 for n, s in zip(shape, spacing)]
    ZZ, YY, XX = np.meshgrid(z, y, x, indexing="ij")

    # 背景：一个大椭球模拟腺体 + 噪声
    body = (YY ** 2 / 45 ** 2 + XX ** 2 / 45 ** 2 + ZZ ** 2 / 40 ** 2) <= 1
    img = np.zeros(shape, dtype=np.float32)
    img[body] = 120.0 + rng.normal(0, 15, size=body.sum())
    img += rng.normal(0, 5, size=shape)

    # 病灶：一个偏心的球形强化灶 + 环形强化边缘
    cz, cy, cx = 0.0, 12.0, -8.0
    r = np.sqrt((ZZ - cz) ** 2 + (YY - cy) ** 2 + (XX - cx) ** 2)
    lesion = r <= 9.0
    rim = (r > 6.5) & (r <= 9.0)
    img[lesion] += 90.0
    img[rim] += 60.0

    mask = lesion.astype(np.uint16)
    # 数组轴序为 (z, y, x)，affine 对角即 (spacing_z, spacing_y, spacing_x)
    affine = np.diag([float(spacing[0]), float(spacing[1]), float(spacing[2]), 1.0])

    img_dir = os.path.join(args.out, "imagesTr")
    seg_dir = os.path.join(args.out, "labelsTr")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)

    ip = os.path.join(img_dir, "phantom_0001.nii.gz")
    sp = os.path.join(seg_dir, "phantom_0001.nii.gz")
    nib.save(nib.Nifti1Image(img.astype(np.int16), affine), ip)
    nib.save(nib.Nifti1Image(mask, affine), sp)

    vox = float(np.prod(spacing))
    print(f"影像 {ip}  shape={shape} spacing={spacing}")
    print(f"掩膜 {sp}  前景 {int(mask.sum())} 体素 = {mask.sum()*vox:.1f} mm³")


if __name__ == "__main__":
    main()
