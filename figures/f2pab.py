import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import h5py
import nature_style
from utils import build_long_dataframe
def read_raw_image(h5path, idx, lr_slice_idx, hr_slice_idx):
    with h5py.File(h5path, 'r') as hf:
        input_image = hf['lr'][idx]
        gt_image = hf['hr'][idx]
        print(f"成功加载H5图像: input shape={input_image.shape}, gt shape={gt_image.shape}")
        input_image = input_image[lr_slice_idx,0]
        gt_image = gt_image[hr_slice_idx,0]
    return input_image, gt_image

mm = nature_style.apply_nature_style()

method_order = ['Ours', '3DRCAN', 'RLN', 'SwinIR', 'CARE', 'SRCNN', 'Interpolation']
# Nature-quality palette: warm → cool gradient, left-to-right descending emphasis
# Ours stands out in strong red; baselines transition smoothly to muted grey
palette_dict = {
    'Ours':           '#E64B35',  # Nature Red — best method, bold standout
    '3DRCAN':         '#F39B7F',  # Salmon — strong 3D baseline
    'RLN':            '#E8A838',  # Amber — strong 3D baseline
    'SwinIR':         '#3C5488',  # Teal — 2D baseline
    'CARE':           '#8491B4',  # Slate blue — 2D baseline
    'SRCNN':          '#B0B9D1',  # Light periwinkle — weaker 2D baseline
    'Interpolation':  '#D9D9D9',  # Neutral grey — trivial baseline
}

# 生成绘图用的 DataFrame
folder_name = "ChemDiffuse/3DSR4z_comparision"
file_name = f'../../../../../../../../m-chimera/chimera/nobackup/yongkang/{folder_name}/metrics.h5'
df_1 = build_long_dataframe(file_name,"3DSR")
# 生成绘图用的 DataFrame
folder_name = "ChemDiffuse/BioTISR_spatial_comparision"
file_name = f'../../../../../../../../m-chimera/chimera/nobackup/yongkang/{folder_name}/metrics.h5'
df_2 = build_long_dataframe(file_name,"BioTISR")
df = pd.concat([df_1, df_2])


# --- 3. 初始化画布 ---
# 宽度 180mm，高度设定为 65mm (根据经验，这足够放两排图)
fig = plt.figure(figsize=(180 * mm, 65 * mm))

# --- 4. 核心布局：Master GridSpec ---
# 将画布分为左右两块：左边给图片 (Panel a)，右边给 Barplot (Panel b)
# width_ratios=[1, 2.5] 表示右边宽度是左边的 2.5 倍
gs_master = fig.add_gridspec(1, 2, width_ratios=[1, 2], wspace=0.15)

gs_left = gs_master[0, 0].subgridspec(2, 1, hspace=0.1)

zSR_lr_raw, zSR_hr_raw = read_raw_image("../../../../../../../../m-chimera/chimera/nobackup/yongkang/ChemDiffuse/3DSR4z_comparision/results.h5",4,2,10)

biotisr_lr_raw, biotisr_hr_raw=read_raw_image('../../../../../../../../m-chimera/chimera/nobackup/yongkang/ChemDiffuse/BioTISR_spatial_comparision/results.h5',0,0,0)
# 补全这部分代码：在gs left这部分，每行绘制数据集示例图（low quality, high quality），左侧标注数据集名字（3DSR，BioTISR），右侧标注低质量、高质量
datasets_config = [
    ("3DSR", zSR_lr_raw, zSR_hr_raw),
    ("BioTISR", biotisr_lr_raw, biotisr_hr_raw)
]

for row_idx, (name, img_lr, img_hr) in enumerate(datasets_config):
    # 1. 在当前行 (gs_left[row_idx]) 内部，再创建一个 1行2列 的子网格
    # wspace=0.05 控制左右两张图的紧凑间距
    gs_inner = gs_left[row_idx].subgridspec(1, 2, wspace=0.05)

    # 创建子图对象
    ax_lr = fig.add_subplot(gs_inner[0])
    ax_hr = fig.add_subplot(gs_inner[1])

    # 2. 绘制图片 (Nature 要求：使用 nearest 插值以保留像素真实感)
    # vmin/vmax 建议根据数据类型统一 (如0-1或0-255)，避免亮度不一致
    ax_lr.imshow(img_lr, cmap='gray', interpolation='nearest')
    ax_hr.imshow(img_hr, cmap='gray', interpolation='nearest')

    # 3. 去掉坐标轴刻度 (不仅是 label，连 tick 线也去掉)
    ax_lr.axis('off')
    ax_hr.axis('off')

    # 4. 标注数据集名字 (Dataset Name) - 放在左侧
    # transform=ax_lr.transAxes 意味着坐标 (0,0) 是左下角，(1,1) 是右上角
    # (-0.1, 0.5) 也就是在左图的左边外面一点，垂直居中
    ax_lr.text(-0.1, 0.5, name, transform=ax_lr.transAxes,
               rotation=90, va='center', ha='right',
               fontsize=7, fontweight='bold', color='black')

    # 5. 标注图像类型 (Low-Quality / High-Quality) - 仅在第一行标注
    if row_idx == 0:
        ax_lr.set_title("Low-Quality", fontsize=7,pad=3)
        ax_hr.set_title("High-Quality", fontsize=7,pad=3)

# === 右侧 Panel b: Bar Plots (2行2列) ===
# 使用 subgridspec 在右边的格子里挖 2x2 = 4 个坑
gs_right = gs_master[0, 1].subgridspec(2, 2, hspace=0.4, wspace=0.25)

axes_metrics = []  # 存起来方便后面操作
flat_metrics_list = ['PSNR', 'SSIM', 'MS-SSIM', 'LPIPS']

for i, metric_name in enumerate(flat_metrics_list):
    row = i // 2
    col = i % 2
    ax = fig.add_subplot(gs_right[row, col])
    axes_metrics.append(ax)

    # 筛选数据
    subset = df[df['Metric'] == metric_name]

    #核心绘图：Seaborn Barplot
    #errorbar=None 去掉误差线(如果只是展示均值)，或者 'sd' 展示标准差
    # sns.barplot(data=subset, x='Dataset', y='Value', hue='Method',
    #             ax=ax, palette=palette_dict, errorbar='sd',
    #             #edgecolor='black',
    #             capsize=0.1,  # 给误差线加个精美的“小帽子”
    #             err_kws={'linewidth': 0.75, 'color': 'black'},
    #             hue_order=method_order,linewidth=0.3)
    # y_vals = subset['Value']
    # grouped = subset.groupby('Method')['Value']
    # means = grouped.mean()
    # stds = grouped.std()
    # max_height_with_err = (means + stds).max()
    # min_height_with_err = (means - stds).min()
    # span = max_height_with_err - min_height_with_err
    # assert span!=0
    # new_bottom = min_height_with_err * 0.6
    # new_top = max_height_with_err*1.1
    # ax.set_ylim(bottom=max(0, new_bottom), top=new_top)

    sns.boxplot(  # 或者 sns.violinplot
        data=subset, x='Dataset', y='Value', hue='Method',
        hue_order=method_order, palette=palette_dict,
        ax=ax,
        linecolor='black',
        linewidth=0.7,
        flierprops={'marker': 'o',
                    'markerfacecolor': 'none',
                    "markeredgecolor":"gray",
                    "markersize":1,
                    'markeredgewidth': 0.4,
                    "alpha":0.6},
        width=0.7,  # 箱子宽度
    )


    ax.set_ylabel(metric_name, labelpad=2)
    ax.set_xlabel('')
    ax.tick_params(axis='x', length=0)  # 去掉 X 轴刻度线
    ax.get_legend().remove()

    # Nature 风格：去掉上右边框
    sns.despine(ax=ax)

    # 给第一个子图加 Label 'b'
    if i == 0:
        ax.text(-0.25, 1.05, 'b', transform=ax.transAxes, fontsize=7, fontweight='bold', va='bottom', ha='right')

# === 5. 添加统一图例 (Shared Legend) ===
# 技巧：从第一个 Barplot 里提取句柄和标签
handles, labels = axes_metrics[0].get_legend_handles_labels()
# 把图例放在右侧区域的顶部 (bbox_to_anchor 是关键)
# 这里的坐标是相对于整个 fig 的，或者相对于某个 ax
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.67, 0.98),
           ncol=7, frameon=False, columnspacing=0.6, handletextpad=0.2, fontsize=5.5)

# --- 6. 导出 ---
# 稍微调整下边距，防止字被切掉
plt.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.10)
plt.savefig('../outputs/Figure_2_Panel_ab_new.pdf', dpi=600, transparent=True)
plt.show()