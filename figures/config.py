"""
Figure Configuration for ChemDiffuse3D.

This file contains all paths and settings for generating paper figures.
Edit the paths below to point to your local data files before running
any plotting scripts.
"""

import os

# =============================================================================
# Base Data Paths
# =============================================================================

# Root directory containing all processed data
DATA_ROOT = "<YOUR_DATA_PATH>"

# Processed result files (HDF5 containing model predictions and ground truth)
RESULTS_PATHS = {
    "3dsr4z": {
        "test": os.path.join(DATA_ROOT, "3dsrnew_data/test/data_d4.h5"),
        "val": os.path.join(DATA_ROOT, "3dsrnew_data/val/data_d4.h5"),
    },
    "3ddenoise": {
        "test": os.path.join(DATA_ROOT, "3ddenoise_data/test/data.h5"),
        "val": os.path.join(DATA_ROOT, "3ddenoise_data/val/data.h5"),
    },
    "biotisr_spatial": {
        "test": os.path.join(DATA_ROOT, "biotisr_data/test/data_spatial.h5"),
        "val": os.path.join(DATA_ROOT, "biotisr_data/val/data_spatial.h5"),
    },
}

# =============================================================================
# Result Keys (names of datasets stored in the HDF5 files)
# =============================================================================

RESULT_KEYS = {
    "chemdiffuse3d": "chemdiffuse3d_output",
    "chemdiffuse3d_rcan": "chemdiffuse3d_output_rcan",
    "chemdiffuse3d_fused": "chemdiffuse3d_output_fused",
    "ground_truth_hr": "hr_cube",
    "ground_truth_hr_denoise": "hr_denoise_cube",
    "low_resolution": "lr_cube",
}

# =============================================================================
# Baseline Result Keys (stored by baseline model scripts)
# =============================================================================

BASELINE_KEYS = {
    "CARE": "care_output",
    "3DRCAN": "rcan_output",
    "SwinIR": "swinir_output",
    "SRCNN": "srcnn_output",
}

# =============================================================================
# Figure Output
# =============================================================================

FIGURE_OUTPUT_DIR = "./figures/output"
os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Plot Styling
# =============================================================================

# Color palette for methods
METHOD_COLORS = {
    "ChemDiffuse3D": "#2196F3",       # Blue
    "ChemDiffuse3D+RCAN": "#1565C0",  # Dark Blue
    "ChemDiffuse3D+Fused": "#0D47A1", # Darker Blue
    "CARE": "#FF9800",                 # Orange
    "3DRCAN": "#4CAF50",               # Green
    "SwinIR": "#9C27B0",               # Purple
    "SRCNN": "#F44336",                # Red
    "Input": "#757575",                # Gray
    "Ground Truth": "#000000",         # Black
}

# Method display order for plots
METHOD_ORDER = [
    "Input",
    "SRCNN",
    "CARE",
    "3DRCAN",
    "SwinIR",
    "ChemDiffuse3D",
    "ChemDiffuse3D+RCAN",
    "ChemDiffuse3D+Fused",
    "Ground Truth",
]
