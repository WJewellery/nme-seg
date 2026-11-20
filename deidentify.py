"""把一个 nnU-Net 训练病例全量脱敏：文件名 + NIfTI 头 + gzip 元数据 + sidecar。"""
import os, sys, json, gzip, shutil, subprocess
import numpy as np
import nibabel as nib

SRC_IMG = sys.argv[1]
SRC_SEG = sys.argv[2]
OUT_IMG = sys.argv[3]
OUT_SEG = sys.argv[4]
NEUTRAL = sys.argv[5]

def rebuild(src_path, dst_path):
    img = nib.load(src_path)
    data = np.asanyarray(img.dataobj)
    # 全新 header：不继承原头任何自由文本/扩展字段
    out = nib.Nifti1Image(data.astype(img.get_data_dtype()), img.affine)
    out.header.set_data_dtype(img.get_data_dtype())
    for f in ['descrip', 'aux_file', 'db_name', 'intent_name']:
        out.header[f] = b''
    out.header['intent_code'] = 0
    out.header['slice_code'] = 0
    out.header['dim_info'] = 0
    out.header['qform_code'] = 1
    out.header['sform_code'] = 1
    if 'glmax' in out.header: out.header['glmax'] = 0
    if 'glmin' in out.header: out.header['glmin'] = 0
    out.header.extensions.clear()
    tmp = dst_path + '.tmp.nii'
    nib.save(out, tmp)
    # gzip -n : 去掉原始文件名和时间戳
    subprocess.run(['gzip', '-n', '-9', '-f', tmp], check=True)
    shutil.move(tmp + '.gz', dst_path)
    return {
        'shape': list(out.shape),
        'voxel_size_mm': [round(float(z), 6) for z in out.header.get_zooms()],
        'dtype': str(out.get_data_dtype()),
        'nonzero_voxels': int((data > 0).sum()),
    }

info_img = rebuild(SRC_IMG, OUT_IMG)
info_seg = rebuild(SRC_SEG, OUT_SEG)

meta = {
    'case_id': NEUTRAL,
    'role': 'nnU-Net 乳腺 MR 肿块分割 训练集样例（已脱敏）',
    'images': f'images/{os.path.basename(OUT_IMG)}',
    'labels': f'labels/{os.path.basename(OUT_SEG)}',
    'image_info': info_img,
    'label_info': info_seg,
    'deidentification': [
        '文件名替换为中性 ID（不含原始病例号/姓名/MR 号）',
        'NIfTI header 重建：descrip/aux_file/db_name/intent_name 清空，intent_code/slice_code/dim_info 归零',
        'NIfTI extensions 全部丢弃（原文件本无扩展）',
        'gzip 用 -n 写入：去掉 FNAME 原始文件名与 mtime 时间戳',
        '无 DICOM/sidecar JSON 随附',
    ],
    'retained': ['体素间距', 'affine（保持几何，便于与原 GT 对齐）', '像素值（未做强度归一化）'],
}
with open(os.path.join(os.path.dirname(os.path.dirname(OUT_IMG)), 'meta.json'), 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(json.dumps(meta, ensure_ascii=False, indent=2))
