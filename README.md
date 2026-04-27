# ChemDiffuse3D

**A Foundation Model for Volumetric Microscopy Image Restoration**

This repository contains the source code accompanying the following manuscript in preparation:

> **ChemDiffuse3D: A Foundation Model for 3D Microscopy Imaging Restoration**
> Yongkang Li et al.

ChemDiffuse3D is a conditional diffusion model for volumetric (3D) microscopy image restoration. It addresses multiple restoration tasks within a single unified framework:

- **3D Super-Resolution** — Recovers high-resolution lateral and axial information from sparsely-sampled Z-stacks (e.g., 4× lateral & 4× axial)
- **3D Denoising** — Removes noise from low signal-to-noise ratio (SNR) volumetric acquisitions
- **Joint Degradation Restoration** — Jointly addresses coupled degradation of image quality and resolution

## Architecture

ChemDiffuse3D combines:
1. **SiT-3D Backbone** — A 3D Diffusion Transformer (Scalable Interpolant Transformer) with Anisotropic Lateral-Axial Attention for efficient processing of volumetric data
2. **3D UNet Conditioning Module** — Encodes low-resolution input volumes into dense conditioning signals
3. **Post-Diffusion Decoders** — RCAN and Fused decoder variants for pixel-space refinement
4. **REPA Alignment (optional)** — REPresentation Alignment loss using DINOv2 features for improved generation quality. This technique is not used in the paper, but in our initial experiments, it accelerates the convergence speed.

## Installation
### Step 1: Clone the repository
```bash
git clone https://github.com/SocraLee/ChemDiffuse3D.git
cd ChemDiffuse3D
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
conda create -n chemdiffuse3d python=3.10 -y
conda activate chemdiffuse3d
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Multi-GPU Training

```bash
accelerate config
```

---

## Demo

A minimal demonstration to verify correct installation and reproduce inference results.

### Demo Data

A small example dataset (`demo_data.h5`) will be made available upon publication at [Zenodo link TBD]. This file contains a subset of pre-processed test volumes with precomputed VAE latents and DINO embeddings.

### Running the Demo

```bash
CUDA_VISIBLE_DEVICES=0 python chemdiffuse3d/generate.py \
    --task_configs_json ./configs/3dsr4z_config.json \
    --backbone_args_json ./configs/backbone_config.json \
    --conditioning_args_json ./configs/encoder_config.json \
    --ckpt_dir <PATH_TO_PRETRAINED_CHECKPOINT> \
    --resume-steps 400000 \
    --exp-name demo_eval
```

### Expected Output

- Predicted high-resolution volumes are saved into the input HDF5 file under the key `chemdiffuse3d_output`
- Console output reports per-task PSNR and SSIM metrics

**Expected runtime:** ~5 minutes for a single test volume on one NVIDIA A100 GPU.

---

## Reproducing Paper Results

### Step 1: Data Preparation

Training data should be organized as HDF5 files with the following structure:

| Key                            | Shape                  | Description                                           |
| ------------------------------ | ---------------------- | ----------------------------------------------------- |
| `hr_cube` or `hr_denoise_cube` | `(N, D, H, W)`         | High-resolution ground truth volumes                  |
| `lr_cube`                      | `(N, D_lr, H, W)`      | Low-resolution input volumes                          |
| `vae_hr_cube`                  | `(N, D, 8, 32, 32)`    | Precomputed VAE latents (mean + std)                  |
| `dino_lr_cube`                 | `(N, D_lr, 256, 1024)` | Precomputed DINOv2 embeddings of LR slices            |
| `dino_hr_cube`                 | `(N, D, 256, 1024)`    | Precomputed DINOv2 embeddings of HR slices (for REPA) |

Use the preprocessing script to extract VAE and DINO features from raw volumes:

```bash
python data_processing/prepare_features.py \
    --data_path <path_to_your_h5_file>
```

Update the data paths in the task configuration files under `configs/` by replacing `<YOUR_DATA_PATH>` with your local data directory.

### Step 2: Training

**Single-task training (e.g., 3D Super-Resolution 4×):**

```bash
accelerate launch \
    --gpu_ids 0,1 \
    --num_processes 2 \
    chemdiffuse3d/train.py \
    --output-dir ./outputs \
    --exp-name 3dsr4z_experiment \
    --backbone_args_json ./configs/backbone_config.json \
    --task_configs_json ./configs/3dsr4z_config.json \
    --conditioning_args_json ./configs/encoder_config.json
```

**Multi-task joint training:**

Use `configs/task_config.json` to define multiple tasks with sampling weights.

**With REPA loss (optional):**

Append `--proj-coeff 0.5 --z-types dinov2 --z-weights 1.0` to enable representation alignment. Note: REPA is not used for the main results in the paper but accelerates convergence in our experiments.

See `scripts/train.sh` for additional examples.

**Expected training time:** ~72 hours for 400K steps on 2× NVIDIA A100 GPUs.

### Step 3: Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python chemdiffuse3d/generate.py \
    --task_configs_json ./configs/3dsr4z_config.json \
    --backbone_args_json ./configs/backbone_config.json \
    --conditioning_args_json ./configs/encoder_config.json \
    --ckpt_dir ./outputs/<exp_name>/checkpoints \
    --resume-steps <step> \
    --exp-name <exp_name>
```

### Step 4: Post-Diffusion Decoder (Optional)

Train and apply a post-diffusion decoder for enhanced pixel-space refinement:

```bash
# Train an RCAN decoder
python chemdiffuse3d/train_decoder.py \
    --task_configs_json configs/3dsr4z_config.json \
    --exp-name 3dsr4z_rcan \
    --save-dir ./outputs/decoder

# Evaluate with the trained decoder
CUDA_VISIBLE_DEVICES=0 python chemdiffuse3d/generate.py \
    --task_configs_json ./configs/3dsr4z_config.json \
    --backbone_args_json ./configs/backbone_config.json \
    --conditioning_args_json ./configs/encoder_config.json \
    --ckpt_dir ./outputs/<exp_name>/checkpoints \
    --resume-steps <step> \
    --exp-name eval_rcan \
    --decoder-type rcan \
    --decoder-path ./outputs/decoder/rcan/3d_sr/best_decoder.pth
```

See `scripts/eval.sh` and `scripts/train_decoder.sh` for additional examples.

### Step 5: Reproducing Figures

Figure reproduction scripts are provided in `figures/`. Edit `figures/config.py` to set the correct paths to your result files, then run individual figure scripts:

```bash
python figures/f2pab.py   # Figure 2 panels a,b
python figures/f3pab.py   # Figure 3 panels a,b
```

---

## Architecture Overview

ChemDiffuse3D integrates the following components:

1. **SiT-3D Backbone** — A 3D Diffusion Transformer (Scalable Interpolant Transformer) with Anisotropic Lateral-Axial Attention for efficient processing of volumetric data
2. **3D UNet Conditioning Module** — Encodes low-resolution input volumes into dense conditioning signals for the diffusion backbone
3. **Post-Diffusion Decoders** — RCAN and Fused decoder variants for pixel-space refinement from VAE latent space
4. **REPA Alignment (optional)** — REPresentation Alignment loss using DINOv2 features for improved generation quality. This technique is not used in the paper, but in our initial experiments it accelerates the convergence speed.

---

## Configuration

### Model Configuration

| File                            | Description                                                       |
| ------------------------------- | ----------------------------------------------------------------- |
| `configs/backbone_config.json`  | SiT-3D backbone hyperparameters (hidden size, depth, heads, etc.) |
| `configs/encoder_config.json`   | LR conditioning encoder parameters                                |
| `configs/3dsr4z_config.json`    | 3D super-resolution (4× Z) task setup                             |
| `configs/3ddenoise_config.json` | 3D denoising task setup                                           |
| `configs/biotisr_config.json`   | Joint degradation restoration task setup                          |
| `configs/task_config.json`      | Multi-task joint training setup                                   |

### Data Paths

Update the `train_data_dir` and `dev_data_dir` fields in the task config JSON files to point to your HDF5 data files. Replace `<YOUR_DATA_PATH>` with your actual data directory.

---

## Project Structure

```
ChemDiffuse3D/
├── chemdiffuse3d/               # Main Python package
│   ├── train.py                 # Multi-task distributed training
│   ├── generate.py              # Inference and evaluation
│   ├── train_decoder.py         # Post-diffusion decoder training
│   ├── loss.py                  # SILoss (denoising + REPA)
│   ├── samplers.py              # Euler & Euler-Maruyama SDE/ODE samplers
│   ├── model/
│   │   ├── model.py             # JointModel top-level wrapper
│   │   ├── sit.py               # Timestep embedder, condition fuser, projectors
│   │   ├── sit_backbone.py      # SiT-3D backbone with factorized attention
│   │   ├── attention.py         # Anisotropic lateral-axial attention
│   │   ├── unet3d.py            # LR conditioning encoder-decoder
│   │   ├── decoder.py           # RCAN & Fused post-diffusion decoders
│   │   ├── utils.py             # Positional embeddings (1D/2D/3D sinusoidal)
│   │   └── rope.py              # Rotary position embeddings
│   └── data/
│       ├── dataset.py           # HDF5-based dataset classes
│       ├── dataloader.py        # Multi-task weighted data loading
│       └── data_utils.py        # PSF simulation and degradation utilities
├── configs/                     # Model and task configuration files
├── scripts/                     # Example shell scripts for training and evaluation
├── data_processing/             # Data preprocessing utilities
├── figures/                     # Figure reproduction scripts
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
└── README.md                    # This file
```

---

## Pre-trained Models

Pre-trained model checkpoints are available at: [Zenodo/HuggingFace link TBD].

*(Links will be updated upon publication.)*

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{li2026chemdiffuse3d,
  title   = {ChemDiffuse3D: A Foundation Model for 3D Microscopy Imaging Restoration},
  author  = {Yongkang Li et al.},
  journal = {Nature Methods},
  year    = {2026}
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

This work builds upon the following open-source projects:

- [Scalable Interpolant Transformers (SiT)](https://github.com/willisma/SiT)
- [Stable Diffusion VAE](https://github.com/CompVis/stable-diffusion)
- [DINOv2](https://github.com/facebookresearch/dinov2)

---

## Contact

For questions or issues regarding this code, please open a [GitHub Issue](https://github.com/SocraLee/ChemDiffuse3D/issues) or contact the corresponding author.
