"""Figure 1 Panels c,d — Z-slice comparison and structural auto-correlation."""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import nature_style
import h5py
import torch
import torch.nn.functional as F

mm = nature_style.apply_nature_style()

# Data pipeline with caching
cache_file = 'auto_ssim_cache.csv'
read_cache = True

if os.path.exists(cache_file) and read_cache:
    df_bell = pd.read_csv(cache_file)
    data = h5py.File('/m-chimera/chimera/nobackup/yongkang/ChemDiffuse/3DSR4z_comparision/results.h5', 'r')
    hr = torch.from_numpy(data['hr'][:]).squeeze()
    lr = torch.from_numpy(data['lr'][:]).squeeze()
    if hr.shape != lr.shape:
        inputs = lr.unsqueeze(1)
        lr = F.interpolate(inputs, size=(hr.shape[1], 256, 256), mode='trilinear', align_corners=False).squeeze(1)
else:
    data = h5py.File('/m-chimera/chimera/nobackup/yongkang/ChemDiffuse/3DSR4z_comparision/results.h5', 'r')
    hr = torch.from_numpy(data['hr'][:]).squeeze()
    lr = torch.from_numpy(data['lr'][:]).squeeze()
    if hr.shape != lr.shape:
        inputs = lr.unsqueeze(1)
        lr = F.interpolate(inputs, size=(hr.shape[1], 256, 256), mode='trilinear', align_corners=False).squeeze(1)

    def compute_bell_curve_data(vol_tensor, distance_scale=1.0):
        n_samples, depth, h, w = vol_tensor.shape
        mid_z = depth // 2
        records = []
        for idx in tqdm(range(n_samples), desc="Computing SSIM"):
            ref_img = vol_tensor[idx, mid_z].numpy()
            for z in range(depth):
                current_img = vol_tensor[idx, z].numpy()
                score = ssim(ref_img, current_img, data_range=1.0)
                dist = (z - mid_z) * distance_scale
                records.append({'Distance': dist, 'SSIM': score})
        return pd.DataFrame(records)

    df_hr = compute_bell_curve_data(hr)
    df_hr['Type'] = 'Target (Ground Truth)'
    df_lr = compute_bell_curve_data(lr)
    df_lr['Type'] = 'Trilinear Interpolation'
    df_bell = pd.concat([df_hr, df_lr])
    df_bell.to_csv(cache_file, index=False)

# Figure layout
fig = plt.figure(figsize=(180 * mm, 65 * mm))
gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.15)

# Panel c: Qualitative Z-slice comparison
gs_c = gs[0].subgridspec(2, 5, wspace=0.05, hspace=0.1)

demo_idx = 1181
center_idx = 10
slice_offsets = [-8, -4, 0, 4, 8]
labels = [r'$z_0 - 8\mu m$', r'$z_0 - 4\mu m$', r'$z_0$', r'$z_0 + 4\mu m$', r'$z_0 + 8\mu m$']
rows_data = [('Trilinear', lr), ('Target', hr)]

for row_i, (name, tensor_data) in enumerate(rows_data):
    for col_i, offset in enumerate(slice_offsets):
        ax = fig.add_subplot(gs_c[row_i, col_i])
        z_idx = int(center_idx + offset)
        z_idx = max(0, min(tensor_data.shape[1] - 1, z_idx))
        img = tensor_data[demo_idx, z_idx].numpy()
        img_h, img_w = img.shape
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')

        if row_i == 0:
            color = 'red' if offset == 0 else 'black'
            weight = 'bold' if offset == 0 else 'normal'
            ax.set_title(labels[col_i], fontsize=7, color=color, fontweight=weight, pad=3)
        if col_i == 0:
            ax.text(-0.1, 0.5, name, transform=ax.transAxes, rotation=90,
                    va='center', ha='right', fontsize=7, fontweight='bold')

        # Highlight center slice with red border
        if offset == 0:
            rect = patches.Rectangle((0, 0), img_w - 1, img_h - 1,
                                     linewidth=2, edgecolor='red', facecolor='none', zorder=10)
            ax.add_patch(rect)

# Panel d: Structural auto-correlation curve
gs_d = gs[1].subgridspec(2, 1, height_ratios=[1, 0.2], hspace=0)
ax_d = fig.add_subplot(gs_d[0])

sns.lineplot(
    data=df_bell, x='Distance', y='SSIM', hue='Type',
    palette={'Target (Ground Truth)': '#003366', 'Trilinear Interpolation': '#D55E00'},
    linewidth=1.5, ax=ax_d,
    errorbar=('sd')
)

ax_d.set_xlabel(r'Relative Distance $\Delta z$ ($\mu m$)', fontsize=7, labelpad=2)
ax_d.set_ylabel('Structural Auto-correlation (SSIM)', fontsize=7, labelpad=2)
ax_d.set_xlim(-10, 10)
ax_d.set_xticks([-8, -4, 0, 4, 8])
ax_d.set_ylim(0, 1.05)
ax_d.legend(frameon=False, fontsize=6, loc='lower center')
ax_d.text(-0.15, 1.1, 'd', transform=ax_d.transAxes, fontsize=8, fontweight='bold')
sns.despine(ax=ax_d)

plt.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.20)
plt.savefig('../outputs/Figure_1_Panel_cd.pdf', dpi=600, transparent=True)
plt.show()