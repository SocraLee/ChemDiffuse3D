import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import numpy as np
import h5py
import torch
import torch.nn.functional as F
import nature_style

# --- 1. 配置 Nature 风格 ---
mm = nature_style.apply_nature_style()
plt.rcParams['mathtext.default'] = 'regular'


# --- 2. 数据读取与处理 ---
def get_image(idx=43):
    # 请确保路径正确
    data_dir = '../../../../../../../../m-chimera/chimera/nobackup/yongkang/ChemDiffuse/3DDenoise_comparision/results.h5'
    data = h5py.File(data_dir, 'r')

    # 读取数据
    hr = torch.from_numpy(data['hr'][idx]).squeeze()
    lr = torch.from_numpy(data['lr'][idx]).squeeze()
    sit = torch.from_numpy(data['sit_pretrain_output_rcan'][idx]).squeeze()
    swinir = torch.from_numpy(data['RCAN_output'][idx]).squeeze()
    target_data = hr.numpy()
    trilinear_data = lr.numpy()
    sit_data = sit.numpy()
    swinir_data = swinir.numpy()
    return target_data, trilinear_data, sit_data, swinir_data

target_data, trilinear_data, sit_data, swinir_data = get_image()

imgs = {
    'Inputs': trilinear_data,
    '3DRCAN': swinir_data,
    'Ours': sit_data,
    'Target': target_data,
}

rows = ['Inputs', '3DRCAN', 'Ours', 'Target']
cols_labels = [r'$z_0$', r'$z_0 + 4\mu m$', r'$z_0 + 8\mu m$', r'$z_0 + 12\mu m$', r'$z_0 + 16\mu m$']

# --- 3. 定义切割位置 (Cut Line) ---
roi_x, roi_y = 90, 5
roi_w, roi_h = 90, 90
highlight_slice = 16  # z_0 + 16 \mu m

cut_y = 60
line_color = 'red'  # 切割线和边框颜色

# --- 4. 初始化画布 ---
fig = plt.figure(figsize=(180 * mm, 120 * mm))
# 调整比例：右侧现在展示侧视图，可以稍微窄一点或保持 3:1
gs_master = fig.add_gridspec(1, 2, width_ratios=[3.2, 0.8], wspace=0.1)

# === 左侧：Main Grid ===
gs_left = gs_master[0, 0].subgridspec(4, 5, hspace=0.05, wspace=0.05)
imshow_args = {'cmap': 'gray', 'vmin': 0, 'vmax': 1, 'interpolation': 'nearest'}

# 我们选择在第 3 列 (Index 2, 即 Depth z+8) 画切割指示线
highlight_col = 2

for r, method in enumerate(rows):
    for i,c in enumerate(range(0,20,4)):
        ax = fig.add_subplot(gs_left[r, i])
        # 获取图像 (Z, H, W) -> 取第 c 层
        # 注意：imgs[method] 是 3D 数组 (D, H, W)
        # 如果 c 超过了数据的深度，需要做保护
        img_3d = imgs[method]
        if c < img_3d.shape[0]:
            img_2d = img_3d[c]
            ax.imshow(img_2d, **imshow_args)

        ax.axis('off')

        if r == 0:
            ax.set_title(cols_labels[i], fontsize=7, pad=3)
        if i == 0:
            ax.text(-0.1, 0.5, method, transform=ax.transAxes,
                    rotation=90, va='center', ha='right', fontsize=7, fontweight='bold')
                    
        # Add scale bar to Target image
        if method == 'Target' and i == 0:
            um_per_px_xy = 301.176 / 512.0
            scalebar_um = 50
            scalebar_px = scalebar_um / um_per_px_xy
            x_start = 10
            y_start = 256 - 15
            rect = patches.Rectangle((x_start, y_start), scalebar_px, 4,
                                      linewidth=0, facecolor='white', zorder=10)
            ax.add_patch(rect)
            ax.text(x_start + scalebar_px / 2, y_start - 3, rf'{scalebar_um} $\mu$m',
                    color='white', fontsize=7, ha='center', va='bottom', fontweight='bold', zorder=10)

        # === 绘制切割指示线 (Indication Line) ===
        ax.axhline(y=cut_y, color=line_color, linestyle='--', linewidth=1, alpha=0.8)

# === 右侧：X-Z Slices (Orthogonal Views) ===
gs_right = gs_master[0, 1].subgridspec(4, 1, hspace=0.1)

methods_xz = ['Inputs', '3DRCAN', 'Ours', 'Target']
z_stretch = 4.0

for i, method in enumerate(methods_xz):
    ax_xz = fig.add_subplot(gs_right[i])

    vol = imgs[method]
    img_xz = vol[:, cut_y, :]

    aspect_xz = (1.0 / (301.176 / 512.0)) * z_stretch
    ax_xz.imshow(img_xz, cmap='gray', aspect=aspect_xz, interpolation='nearest', vmin=0, vmax=1)

    if i == 0:
        ax_xz.set_title('X-Z Slice', fontsize=7)

    if i == 1:
        ax_xz.set_ylabel(f'z-axis ({int(z_stretch)}x)', fontsize=6, labelpad=1)
        ax_xz.yaxis.set_label_coords(-0.1, -0.1)

    if i == len(methods_xz) - 1:
        ax_xz.set_xlabel('x-axis', fontsize=6, labelpad=1)

    ax_xz.set_xticks([])
    ax_xz.set_yticks([])

    for spine in ax_xz.spines.values():
        spine.set_edgecolor(line_color)
        spine.set_linewidth(1.5)

# --- 5. 导出 ---
# 调整布局防止标签重叠
plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08)
plt.savefig('../outputs/Figure_3_Panel_c.pdf', dpi=600, transparent=True)
print("Saved: ../outputs/Figure_3_Panel_c.pdf")
plt.close(fig)

# ══════════════════════════════════════════════════════════════
# Supplementary Figure: Sup_Fig_3.pdf
# Row 1: Full patches at highlight_slice for all 4 methods
# Row 2: Zoom-in patches with ROI boxes and connection lines
# ══════════════════════════════════════════════════════════════
from matplotlib.patches import ConnectionPatch
sup_methods = ['Inputs', '3DRCAN', 'Ours', 'Target']

# Get images specific for the supplementary figure (idx=63)
target_data_sup, trilinear_data_sup, sit_data_sup, swinir_data_sup = get_image(63)
imgs_sup = {
    'Inputs': trilinear_data_sup,
    '3DRCAN': swinir_data_sup,
    'Ours': sit_data_sup,
    'Target': target_data_sup,
}

fig_sup = plt.figure(figsize=(130 * mm, 70 * mm))
gs_sup = fig_sup.add_gridspec(2, 4, hspace=0.15, wspace=0.05)

ax_full = []  # Store row-1 axes for connection lines
ax_zoom = []  # Store row-2 axes for connection lines

for col, method in enumerate(sup_methods):
    # Row 1: Full patch
    ax_f = fig_sup.add_subplot(gs_sup[0, col])
    img_full = imgs_sup[method][highlight_slice]
    ax_f.imshow(img_full, **imshow_args)
    ax_f.set_title(method, fontsize=7, fontweight='bold', pad=3)
    ax_f.axis('off')

    # Draw ROI box on full patch
    rect = patches.Rectangle((roi_x, roi_y), roi_w, roi_h,
                              linewidth=1.5, edgecolor='red', facecolor='none', linestyle='--')
    ax_f.add_patch(rect)
    ax_full.append(ax_f)

    if method == 'Inputs':
        um_per_px_xy = 301.176 / 512.0
        scalebar_um_f = 50
        scalebar_px_f = scalebar_um_f / um_per_px_xy
        x_start_f = 10
        y_start_f = 256 - 15  # Assume 256x256
        rect_f = patches.Rectangle((x_start_f, y_start_f), scalebar_px_f, 4,
                                   linewidth=0, facecolor='white', zorder=50)
        ax_f.add_patch(rect_f)
        ax_f.text(x_start_f + scalebar_px_f / 2, y_start_f - 3, rf'{scalebar_um_f} $\mu$m',
                  color='white', fontsize=7, ha='center', va='bottom', fontweight='bold', zorder=50)

    # Row 2: Zoom-in patch
    ax_z = fig_sup.add_subplot(gs_sup[1, col])
    zoom_img = img_full[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    ax_z.imshow(zoom_img, **imshow_args)
    ax_z.set_xticks([])
    ax_z.set_yticks([])

    for spine in ax_z.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('red')
        spine.set_linewidth(1.5)
        spine.set_linestyle('-')

    if method == 'Inputs':
        scalebar_um_z = 10
        scalebar_px_z = scalebar_um_z / um_per_px_xy
        x_start_z = 5
        y_start_z = roi_h - 10
        rect_z = patches.Rectangle((x_start_z, y_start_z), scalebar_px_z, 2,
                                   linewidth=0, facecolor='white', zorder=50)
        ax_z.add_patch(rect_z)
        ax_z.text(x_start_z + scalebar_px_z / 2, y_start_z - 2, rf'{scalebar_um_z} $\mu$m',
                  color='white', fontsize=7, ha='center', va='bottom', fontweight='bold', zorder=50)

    ax_zoom.append(ax_z)

    # Connection lines
    con_args = dict(coordsA="data", coordsB="data",
                    color="red", linestyle="--", linewidth=1.0, alpha=0.9,
                    arrowstyle="-", clip_on=False, zorder=5)

    dest_h, dest_w = zoom_img.shape
    # Line 1: ROI bottom-left -> Zoom top-left
    con1 = ConnectionPatch(xyA=(roi_x, roi_y + roi_h),
                           xyB=(0, 0),
                           axesA=ax_f, axesB=ax_z, **con_args)
    ax_f.add_artist(con1)

    # Line 2: ROI bottom-right -> Zoom top-right
    con2 = ConnectionPatch(xyA=(roi_x + roi_w, roi_y + roi_h),
                           xyB=(dest_w - 0.5, 0),
                           axesA=ax_f, axesB=ax_z, **con_args)
    ax_f.add_artist(con2)

plt.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.05)
plt.savefig('../outputs/Sup_Fig_3.pdf', dpi=600, transparent=True)
print("Saved: ../outputs/Sup_Fig_3.pdf")
plt.close(fig_sup)

print("Done!")