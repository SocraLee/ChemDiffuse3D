# Figure Reproduction Scripts

This directory contains the plotting scripts used to generate figures for the paper.

## Setup

Before running any scripts, edit `config.py` to set the correct data paths for your environment:

```python
DATA_ROOT = "/path/to/your/processed/data"
```

## Scripts

| Script | Description |
|--------|-------------|
| `f1pab.py` | Figure 1 panels a,b — Overview visualizations |
| `f1pd.py` | Figure 1 panel d — Detailed comparison |
| `f1_cube_visual.py` | Figure 1 — 3D cube rendering |
| `f2pab.py` | Figure 2 panels a,b — Quantitative metrics |
| `f2pc.py` | Figure 2 panel c — Sample visualizations |
| `f3pab.py` | Figure 3 panels a,b — Denoising results |
| `f3pc.py` | Figure 3 panel c — Denoising visualizations |
| `f4pa.py` | Figure 4 panel a — Ablation study |
| `f4pa_new.py` | Figure 4 panel a — Updated ablation |
| `f4pcd.py` | Figure 4 panels c,d — Correlation analysis |
| `supf6.py` | Supplementary Figure 6 |
| `supf7.py` | Supplementary Figure 7 |
| `supf8.py` | Supplementary Figure 8 |

## Dependencies

- `matplotlib`
- `numpy`
- `h5py`
- `scipy`

## Note

These scripts reference result files (`.h5`) that contain both model predictions
and ground truth data. The paths in individual scripts should be updated to use
the centralized `config.py` configuration. A `nature_style.py` module provides
consistent styling for Nature sub-journal formatting.
