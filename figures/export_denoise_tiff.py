"""Export specific volumes from Denoise results.h5 as per-slice PNG files."""
import os
import numpy as np
import h5py
import imageio.v3 as iio

H5_PATH = '/m-chimera/chimera/nobackup/yongkang/ChemDiffuse/3DDenoise_comparision/results.h5'
OUT_DIR = '../outputs/denoise_png'
INDICES = [919, 1281]
KEYS = {'RCAN_output': '3DRCAN', 'sit_pretrain_output_rcan': 'Ours'}

os.makedirs(OUT_DIR, exist_ok=True)

with h5py.File(H5_PATH, 'r') as f:
    for idx in INDICES:
        for h5_key, name in KEYS.items():
            vol = np.squeeze(f[h5_key][idx]).astype(np.float32)
            # normalise to 8-bit per volume for consistent contrast across slices
            lo, hi = float(vol.min()), float(vol.max())
            vol_u8 = ((vol - lo) / max(hi - lo, 1e-12) * 255.0).clip(0, 255).astype(np.uint8)
            sub_dir = f'{OUT_DIR}/{name}_idx{idx}'
            os.makedirs(sub_dir, exist_ok=True)
            for z in range(vol_u8.shape[0]):
                out_path = f'{sub_dir}/slice_{z:03d}.png'
                iio.imwrite(out_path, vol_u8[z])
            print(f'{name}_idx{idx}: {vol_u8.shape[0]} slices -> {sub_dir}')

print('Done!')
