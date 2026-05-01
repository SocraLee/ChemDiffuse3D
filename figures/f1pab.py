"""Figure 1 Panels a,b — Dataset overview and metric comparison bar chart."""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import h5py
import nature_style

def read_raw_image(h5path, idx, lr_slice_idx, hr_slice_idx):
    with h5py.File(h5path, 'r') as hf:
        input_image = hf['lr'][idx]
        gt_image = hf['hr'][idx]
        input_image = input_image[lr_slice_idx, 0]
        gt_image = gt_image[hr_slice_idx, 0]
    return input_image, gt_image

mm = nature_style.apply_nature_style()

method_order = ['Ours', 'SwinIR', 'CARE', 'SRCNN', 'Interpolation']
colors = [
    '#2f2d54',  # Ours
    '#9193b4',  # SwinIR
    '#bd9aad',  # CARE
    '#9e9e9e',  # SRCNN
    '#e8d2b3'   # Interpolation
]
palette_dict = dict(zip(method_order, colors))

data_1 = {
    "Dataset": ["3DSR"] * 5,
    'Method': ['Interpolation', 'SRCNN', 'CARE', 'SwinIR', 'Ours'],
    'SSIM': [27.47, 36.27, 36.17, 36.71, 38.20],
    'MS-SSIM': [40.55, 40.87, 41.89, 40.54, 51.28],
    'PSNR': [14.10, 15.82, 15.85, 15.90, 16.63],
    'LPIPS': [52.40, 55.87, 54.53, 54.94, 50.14]
}
df_1 = pd.DataFrame(data_1)
data_2 = {
    "Dataset": ["BioTISR"] * 5,
    'Method': ['Interpolation', 'SRCNN', 'CARE', 'SwinIR', 'Ours'],
    'SSIM': [10.71, 30.29, 28.78, 33.41, 37.56],
    'MS-SSIM': [26.51, 32.65, 37.03, 39.48, 49.92],
    'PSNR': [12.48, 16.58, 16.78, 17.14, 17.62],
    'LPIPS': [61.68, 62.44, 60.00, 57.29, 50.65]
}
df_2 = pd.DataFrame(data_2)
df = pd.concat([df_1, df_2])

df = pd.melt(
    df,
    id_vars=['Dataset', 'Method'],
    value_vars=['SSIM', 'MS-SSIM', 'PSNR', 'LPIPS'],
    var_name='Metric',
    value_name='Value'
)

# Initialize figure
fig = plt.figure(figsize=(180 * mm, 65 * mm))

# Layout: left = sample images (Panel a), right = bar plots (Panel b)
gs_master = fig.add_gridspec(1, 2, width_ratios=[1, 2], wspace=0.15)
gs_left = gs_master[0, 0].subgridspec(2, 1, hspace=0.1)

zSR_lr_raw, zSR_hr_raw = read_raw_image(
    "/m-chimera/chimera/nobackup/yongkang/MicroDiffuse/3DSR4z_comparision/results.h5", 4, 2, 10)
biotisr_lr_raw, biotisr_hr_raw = read_raw_image(
    '/m-chimera/chimera/nobackup/yongkang/MicroDiffuse/BioTISR_spatial_comparision/results.h5', 0, 0, 0)

datasets_config = [
    ("3DSR", zSR_lr_raw, zSR_hr_raw),
    ("BioTISR", biotisr_lr_raw, biotisr_hr_raw)
]

for row_idx, (name, img_lr, img_hr) in enumerate(datasets_config):
    gs_inner = gs_left[row_idx].subgridspec(1, 2, wspace=0.05)
    ax_lr = fig.add_subplot(gs_inner[0])
    ax_hr = fig.add_subplot(gs_inner[1])
    ax_lr.imshow(img_lr, cmap='gray', interpolation='nearest')
    ax_hr.imshow(img_hr, cmap='gray', interpolation='nearest')
    ax_lr.axis('off')
    ax_hr.axis('off')
    ax_lr.text(-0.1, 0.5, name, transform=ax_lr.transAxes,
               rotation=90, va='center', ha='right',
               fontsize=7, fontweight='bold', color='black')
    if row_idx == 0:
        ax_lr.set_title("Low-Quality", fontsize=7, pad=3)
        ax_hr.set_title("High-Quality", fontsize=7, pad=3)

# Right panel: bar plots (2x2 grid)
gs_right = gs_master[0, 1].subgridspec(2, 2, hspace=0.4, wspace=0.25)

axes_metrics = []
flat_metrics_list = ['PSNR', 'SSIM', 'MS-SSIM', 'LPIPS']

for i, metric_name in enumerate(flat_metrics_list):
    row = i // 2
    col = i % 2
    ax = fig.add_subplot(gs_right[row, col])
    axes_metrics.append(ax)

    subset = df[df['Metric'] == metric_name]
    sns.barplot(data=subset, x='Dataset', y='Value', hue='Method',
                ax=ax, palette=palette_dict, errorbar=None, edgecolor='black',
                hue_order=method_order, linewidth=0.3)

    y_vals = subset['Value']
    ax.set_ylim(bottom=y_vals.min() * 0.85, top=y_vals.max() * 1.05)
    ax.set_ylabel(metric_name, labelpad=2)
    ax.set_xlabel('')
    ax.tick_params(axis='x', length=0)
    ax.get_legend().remove()
    sns.despine(ax=ax)

    if i == 0:
        ax.text(-0.25, 1.05, 'b', transform=ax.transAxes, fontsize=7,
                fontweight='bold', va='bottom', ha='right')

# Shared legend
handles, labels = axes_metrics[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.67, 0.98),
           ncol=5, frameon=False, columnspacing=1.0, handletextpad=0.3)

plt.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.10)
plt.savefig('../outputs/Figure_1_Panel_ab.pdf', dpi=600, transparent=True)
plt.show()