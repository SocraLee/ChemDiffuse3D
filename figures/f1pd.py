import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

import nature_style
import h5py
import torch
import torch.nn.functional as F

# --- 1. 配置 ---
mm = nature_style.apply_nature_style()

# --- 2. 核心：带缓存的数据管道 (Data Pipeline with Caching) ---
cache_file = 'auto_ssim_cache.csv'
read_cache = True
# 如果缓存文件存在，直接“秒读”，跳过计算
if os.path.exists(cache_file) and read_cache:
    print(f"Found cache file '{cache_file}'. Loading data directly...")
    df_bell = pd.read_csv(cache_file)

    # 因为我们跳过了加载 h5，所以这里要为了画 Panel C 单独加载一下切片数据
    # 注意：如果全量加载 h5 太慢，你可以把这一步也优化掉，只读需要的 idx
    print("Loading image tensors for Panel C...")
    data = h5py.File('../../../../../../../../m-chimera/chimera/nobackup/yongkang/ChemDiffuse/3DSR4z_comparision/results.h5',
                     'r')
    hr = torch.from_numpy(data['hr'][:]).squeeze()
    lr = torch.from_numpy(data['lr'][:]).squeeze()
    if hr.shape != lr.shape:
        inputs = lr.unsqueeze(1)
        lr = F.interpolate(inputs, size=(hr.shape[1], 256, 256), mode='trilinear', align_corners=False).squeeze(1)

else:
    print("Cache not found. Loading tensors and computing SSIM...")
    data = h5py.File('../../../../../../../../m-chimera/chimera/nobackup/yongkang/ChemDiffuse/3DSR4z_comparision/results.h5',
                     'r')
    hr = torch.from_numpy(data['hr'][:]).squeeze()
    lr = torch.from_numpy(data['lr'][:]).squeeze()

    if hr.shape != lr.shape:
        inputs = lr.unsqueeze(1)
        lr = F.interpolate(inputs, size=(hr.shape[1], 256, 256), mode='trilinear', align_corners=False).squeeze(1)

    def compute_bell_curve_data(vol_tensor, distance_scale=1.0):
        n_samples, depth, h, w = vol_tensor.shape
        mid_z = depth // 2
        records = []
        # 为了看到置信区间，这里至少取 100 个样本（建议最终版填 n_samples）
        sample_indices = range(n_samples)  # 全量计算

        for idx in tqdm(sample_indices, desc="Computing SSIM"):
            ref_img = vol_tensor[idx, mid_z].numpy()
            for z in range(depth):
                current_img = vol_tensor[idx, z].numpy()
                score = ssim(ref_img, current_img, data_range=1.0)
                dist = (z - mid_z) * distance_scale
                records.append({'Distance': dist, 'SSIM': score})
        return pd.DataFrame(records)

    print("Computing Target Bell Curve...")
    df_hr = compute_bell_curve_data(hr)
    df_hr['Type'] = 'Target (Ground Truth)'

    print("Computing Interpolation Bell Curve...")
    df_lr = compute_bell_curve_data(lr)
    df_lr['Type'] = 'Trilinear Interpolation'

    df_bell = pd.concat([df_hr, df_lr])

    # 保存结果到硬盘，下次就不用算了！
    df_bell.to_csv(cache_file, index=False)
    print(f"Data cached successfully to '{cache_file}'.")

# --- 3. 绘图开始 (完美横版布局) ---
# 宽度 180mm，因为是横排，高度不需要太高，大约 55mm 即可
fig = plt.figure(figsize=(180 * mm, 65 * mm))

# 左右分栏：左边切片(Panel c)占 2.2 份宽度，右边折线图(Panel d)占 1 份宽度
gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.15)

# ==========================================
# Panel C: Qualitative Slices (The Visuals)
# ==========================================
# 在左边的大格子里，再分 2行 5列
gs_c = gs[0].subgridspec(2, 5, wspace=0.05, hspace=0.1)

demo_idx = 1181
center_idx = 10  # 真实的物理中间层
slice_offsets = [-8, -4, 0, 4, 8]
labels = [r'$z_0 - 8\mu m$', r'$z_0 - 4\mu m$', r'$z_0$', r'$z_0 + 4\mu m$', r'$z_0 + 8\mu m$']
rows_data = [('Trilinear', lr), ('Target', hr)]

for row_i, (name, tensor_data) in enumerate(rows_data):
    for col_i, offset in enumerate(slice_offsets):
        ax = fig.add_subplot(gs_c[row_i, col_i])

        z_idx = int(center_idx + offset)
        z_idx = max(0, min(tensor_data.shape[1] - 1, z_idx))  # 边界保护

        img = tensor_data[demo_idx, z_idx].numpy()
        # 获取图像宽高，用于画框
        img_h, img_w = img.shape

        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')  # 霸道的 axis('off')

        if row_i == 0:
            color = 'red' if offset == 0 else 'black'
            weight = 'bold' if offset == 0 else 'normal'
            ax.set_title(labels[col_i], fontsize=7, color=color, fontweight=weight, pad=3)

        if col_i == 0:
            ax.text(-0.1, 0.5, name, transform=ax.transAxes, rotation=90,
                    va='center', ha='right', fontsize=7, fontweight='bold')

        # --- 核心修复：用 Rectangle 暴力画框，无视 axis('off') ---
        if offset == 0:
            # 坐标 (0,0) 开始，宽 img_w，高 img_h
            rect = patches.Rectangle((0, 0), img_w - 1, img_h - 1,
                                     linewidth=2, edgecolor='red', facecolor='none', zorder=10)
            ax.add_patch(rect)

# Add Panel Label 'c'
# ==========================================
# Panel D: Quantitative Curve (The Statistics)
# ==========================================
#ax_d = fig.add_subplot(gs[1])
gs_d = gs[1].subgridspec(2, 1, height_ratios=[1, 0.2], hspace=0)
ax_d = fig.add_subplot(gs_d[0])

# --- 核心修复：显式强制开启 95% 置信区间 ---
sns.lineplot(
    data=df_bell, x='Distance', y='SSIM', hue='Type',
    palette={'Target (Ground Truth)': '#003366', 'Trilinear Interpolation': '#D55E00'},
    linewidth=1.5, ax=ax_d,
    errorbar=('sd')  # 强行开启 CI，确保有足够的样本支撑
)

ax_d.set_xlabel(r'Relative Distance $\Delta z$ ($\mu m$)', fontsize=7, labelpad=2)
ax_d.set_ylabel('Structural Auto-correlation (SSIM)', fontsize=7, labelpad=2)

ax_d.set_xlim(-10, 10)
ax_d.set_xticks([-8,-4,0,4,8])  # 横版空间较小，刻度不能太密，步长设为4
ax_d.set_ylim(0, 1.05)
#ax_d.grid(True, linestyle='--', alpha=0.3)

# 调整图例：横排版面下，放到图形内部的右下角或下方
ax_d.legend(frameon=False, fontsize=6, loc='lower center')

# Add Panel Label 'd'
ax_d.text(-0.15, 1.1, 'd', transform=ax_d.transAxes, fontsize=8, fontweight='bold')
sns.despine(ax=ax_d)

# 挤压布局，防止文字被切掉
plt.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.20)
plt.savefig('../outputs/Figure_1_Panel_cd.pdf', dpi=600, transparent=True)
plt.show()