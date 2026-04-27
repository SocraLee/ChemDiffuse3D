import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from tqdm import tqdm


def create_comparison_gif(data, output_filename, titles=None, fps=5):
    if not titles:titles=['LR',"2D Baseline","Ours","HR"]
    frames = []
    depth = data[-1].shape[0]
    fig, axes = plt.subplots(1, 4, figsize=(9, 3.5), dpi=100)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.85, wspace=0.1, hspace=0.1)
    for d in tqdm(range(depth)):
        axes[0].imshow(data[0][d].squeeze(), cmap='gray')
        axes[1].imshow(data[1][d].squeeze(), cmap='gray')
        axes[2].imshow(data[2][d].squeeze(), cmap='gray')
        axes[3].imshow(data[3][d].squeeze(), cmap='gray')
        for i, ax in enumerate(axes):
            ax.set_title(titles[i])
            ax.axis('off')
            ax.set_aspect('equal', adjustable='box')
        fig.suptitle(f'Depth Slice: {d + 1}/{depth}', fontsize=14)
        fig.canvas.draw()
        frame_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame_image = frame_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame_image)
    plt.close(fig)
    print(f"Saving GIF to {output_filename}...")
    imageio.mimsave(output_filename, frames, fps=fps, loop=0)


import pandas as pd
import numpy as np
import h5py

def build_long_dataframe(h5_path, dataset):
    methods = ['Interpolation', 'SRCNN', 'CARE', 'SwinIR', 'RCAN', 'RLN', 'SiTfromPretrainRcan']
    methods_mapping = {m: m for m in methods}
    methods_mapping['SiTfromPretrainRcan'] = "Ours"
    methods_mapping['RCAN'] = "3DRCAN"
    if dataset == "":
        methods_mapping['Interpolation'] = "Inputs"
    metrics = ['ssim', 'psnr', 'msssim', 'lpips']
    metrics_v = ['SSIM', 'PSNR', 'MS-SSIM', 'LPIPS']
    metrics_mapping = {metrics[i]: metrics_v[i] for i in range(4)}

    data_rows = []
    with h5py.File(h5_path, 'r') as f:
        for method in methods:
            for metric in metrics:
                raw_values = f[f"{method}_{metric}"][:]
                for val in raw_values:
                    data_rows.append({
                        'Dataset': dataset,
                        'Method': methods_mapping[method],
                        'Metric': metrics_mapping[metric],
                        'Value': float(val)  # 确保是 float
                    })
    df = pd.DataFrame(data_rows)
    print(df.head(6))
    return df