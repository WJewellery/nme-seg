#!/usr/bin/env python3
"""分割优先的重构流程：掩膜后处理 -> 分割指标 + 病例级检出

与旧代码的区别：
  旧：预测掩膜 -> 算体积 -> 按体积做 ROC 阈值 -> 病例阴阳（掩膜从不裁剪）
  新：预测掩膜 -> 掩膜上剪连通域 -> 用【裁剪后的掩膜】算分割指标
                                  -> 病例级阴阳直接由掩膜是否为空推导
"""
import os, argparse, json
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from skimage.measure import label


def load(path):
    img = nib.load(path)
    return np.asanyarray(img.dataobj), img.header.get_zooms()[:3]


def cc_prune(binary, vox_mm3, min_component_mm3):
    """在掩膜上剪连通域：删除体积 < min_component_mm3 的孤立连通域。"""
    lab, n = label(binary, connectivity=3, return_num=True)
    if n == 0:
        return np.zeros_like(binary), 0, 0, 0.0
    sizes = np.bincount(lab.ravel())
    keep = [i for i in range(1, n + 1) if sizes[i] * vox_mm3 >= min_component_mm3]
    removed_mm3 = sum(sizes[i] * vox_mm3 for i in range(1, n + 1) if i not in keep)
    out = np.isin(lab, keep) if keep else np.zeros_like(binary)
    return out, n, len(keep), float(removed_mm3)


def surface_metrics(pred, gt, spacing):
    """HD95 / ASD（mm），用 SimpleITK 表面距离。"""
    if pred.sum() == 0 or gt.sum() == 0:
        return (0.0, 0.0) if (pred.sum() == 0 and gt.sum() == 0) else (np.nan, np.nan)
    p = sitk.GetImageFromArray(pred.astype(np.uint8)); p.SetSpacing([float(s) for s in spacing])
    g = sitk.GetImageFromArray(gt.astype(np.uint8));  g.SetSpacing([float(s) for s in spacing])
    p_surf = sitk.GetArrayFromImage(sitk.LabelContour(p)) > 0
    g_surf = sitk.GetArrayFromImage(sitk.LabelContour(g)) > 0
    pd = sitk.GetArrayFromImage(sitk.Abs(sitk.SignedMaurerDistanceMap(p, squaredDistance=False, useImageSpacing=True)))
    gd = sitk.GetArrayFromImage(sitk.Abs(sitk.SignedMaurerDistanceMap(g, squaredDistance=False, useImageSpacing=True)))
    if p_surf.sum() == 0 or g_surf.sum() == 0:
        return (np.nan, np.nan)
    d = np.concatenate([gd[p_surf], pd[g_surf]])
    return float(np.percentile(d, 95)), float(np.mean(d))


def lesion_recall(pred, gt):
    """病灶级召回：GT 的每个连通域中，被预测覆盖的比例 >=10% 即算召回。"""
    glab, gn = label(gt, connectivity=3, return_num=True)
    if gn == 0:
        return 1.0
    hit = 0
    for i in range(1, gn + 1):
        comp = glab == i
        if (pred & comp).sum() / comp.sum() >= 0.10:
            hit += 1
    return hit / gn


def case_metrics(pred, gt, spacing):
    vox = float(np.prod(spacing))
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    d = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else (1.0 if (gt.sum() == 0 and pred.sum() == 0) else 0.0)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else (1.0 if (gt.sum() == 0 and pred.sum() == 0) else 0.0)
    sens = tp / (tp + fn) if (tp + fn) else (1.0 if pred.sum() == 0 else 0.0)
    spec = tn / (tn + fp) if (tn + fp) else 1.0
    prec = tp / (tp + fp) if (tp + fp) else (1.0 if gt.sum() == 0 else 0.0)
    npv = tn / (tn + fn) if (tn + fn) else 1.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    tpv = tp / (tp + fp) if (tp + fp) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    gt_mm3, pre_mm3 = gt.sum() * vox, pred.sum() * vox
    hd95, asd = surface_metrics(pred, gt, spacing)
    return dict(TP=tp, FP=fp, TN=tn, FN=fn,
                Dice=round(d, 4), IoU=round(iou, 4), Sensitivity=round(sens, 4),
                Specificity=round(spec, 4), Precision=round(prec, 4), F1=round(f1, 4),
                NPV=round(npv, 4), TPV=round(tpv, 4), ACC=round(acc, 4),
                HD95=round(hd95, 3) if hd95 == hd95 else np.nan,
                ASD=round(asd, 3) if asd == asd else np.nan,
                lesion_recall=round(lesion_recall(pred, gt), 4),
                gt_volume=round(gt_mm3, 2), pre_volume=round(pre_mm3, 2),
                vol_diff=round(pre_mm3 - gt_mm3, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--modes", default="raw,cc250,cc50_case250")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    names = sorted(f for f in os.listdir(args.pred_dir) if f.endswith(".nii.gz"))
    if args.limit:
        names = names[:args.limit]

    modes = args.modes.split(",")
    results = {m: [] for m in modes}

    for i, name in enumerate(names, 1):
        praw, spacing = load(os.path.join(args.pred_dir, name))
        gtarr, _ = load(os.path.join(args.gt_dir, name))
        vox = float(np.prod(spacing))
        pb_raw, gb = praw > 0.5, gtarr > 0.5

        # 三种后处理配置
        pb_cc250, n250, k250, rm250 = cc_prune(pb_raw, vox, 250.0)
        pb_cc50, n50, k50, rm50 = cc_prune(pb_raw, vox, 50.0)
        # cc50_case250: 先 50 去噪点，再若总体积 <250 判为无病灶
        pb_c50c250 = pb_cc50 if (pb_cc50.sum() * vox >= 250.0) else np.zeros_like(pb_cc50)

        variants = dict(raw=pb_raw, cc250=pb_cc250, cc50_case250=pb_c50c250)
        for m in modes:
            r = case_metrics(variants[m], gb, spacing)
            r.update(case=name, gt_positive=int(gb.sum() > 0), pre_positive=int(variants[m].sum() > 0),
                     n_components=int(n250), n_kept=int(k250), removed_mm3=round(rm250, 2))
            results[m].append(r)
        if i % 10 == 0 or i == len(names):
            print(f"  [{i}/{len(names)}] {name}", flush=True)

    json.dump(results, open(args.out_json, "w"), ensure_ascii=False)
    print("saved", args.out_json)


if __name__ == "__main__":
    main()
