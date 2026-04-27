# ChemDiffuse3D

**Conditional 3D Diffusion Model for Volumetric Microscopy Image Restoration**

ChemDiffuse3D is a conditional diffusion model based on flow-matching that performs volumetric (3D) image restoration tasks for microscopy data, including:

- **3D Super-Resolution**: Recovers high-resolution axial information from sparsely-sampled Z-stacks (e.g., 4× Z-upsampling)
- **3D Denoising**: Removes noise from low-SNR volumetric acquisitions
- **Isotropic Restoration (BioTISR)**: Corrects anisotropic resolution to achieve near-isotropic 3D volumes

## Architecture

ChemDiffuse3D combines:
1. **SiT-3D Backbone** — A 3D Scalable Interpolant Transformer with factorized spatial-depth attention for efficient processing of volumetric data
2. **3D UNet Conditioning Module** — Encodes low-resolution input volumes into dense conditioning signals
3. **REPA Alignment** — Optional REPresentation Alignment loss using DINOv2 features for improved generation quality
4. **Post-Diffusion Decoders** — RCAN and Fused decoder variants for pixel-space refinement

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/ChemDiffuse3D.git
cd ChemDiffuse3D

# Install dependencies
pip install -r requirements.txt

# (Optional) Configure HuggingFace Accelerate for multi-GPU training
accelerate config
```

## Data Preparation

### HDF5 Format

Training data should be organized as HDF5 files with the following datasets:

| Key | Shape | Description |
|-----|-------|-------------|
| `hr_cube` or `hr_denoise_cube` | `(N, D, H, W)` | High-resolution volumes |
| `lr_cube` | `(N, D_lr, H, W)` | Low-resolution input volumes |
| `vae_hr_cube` | `(N, D, 8, 32, 32)` | Precomputed VAE latents (mean + std) |
| `dino_lr_cube` | `(N, D_lr, 256, 1024)` | Precomputed DINOv2 embeddings of LR slices |
| `dino_hr_cube` | `(N, D, 256, 1024)` | Precomputed DINOv2 embeddings of HR slices (for REPA) |

### Feature Extraction

Use the preprocessing script to extract VAE and DINO features:

```bash
python data_processing/prepare_features.py \
    --data_path <path_to_your_h5_file>
```

## Training

### Single-Task Training

```bash
accelerate launch \
    --gpu_ids 0,1 \
    --num_processes 2 \
    chemdiffuse3d/train.py \
    --report-to="wandb" \
    --allow-tf32 \
    --mixed-precision="fp16" \
    --seed=0 \
    --path-type="linear" \
    --prediction="v" \
    --weighting="uniform" \
    --proj-coeff=0 \
    --output-dir="./outputs" \
    --exp-name="3dsr4z_experiment" \
    --resume-step 0 \
    --sampling-steps 5000 \
    --checkpointing-steps 10000 \
    --max-train-steps 400000 \
    --gradient-accumulation-steps 5 \
    --backbone_args_json ./configs/backbone_config.json \
    --task_configs_json ./configs/3dsr4z_config.json \
    --conditioning_args_json ./configs/encoder_config.json \
    --project-name "ChemDiffuse3D"
```

### With REPA Loss

Add `--proj-coeff 0.5 --z-types dinov2 --z-weights 1.0` to enable representation alignment.

### Multi-Task Joint Training

Use `configs/task_config.json` to define multiple tasks with sampling weights.

See `scripts/train.sh` for more examples.

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python chemdiffuse3d/generate.py \
    --task_configs_json "./configs/3dsr4z_config.json" \
    --backbone_args_json "./configs/backbone_config.json" \
    --conditioning_args_json "./configs/encoder_config.json" \
    --ckpt_dir "./outputs/3dsr4z_experiment/checkpoints" \
    --resume-steps 40000 \
    --exp-name "eval_40k" \
    --save-results \
    --batch-size 4 \
    --num-steps 50 \
    --results-key chemdiffuse3d_output
```

### With Post-Diffusion Decoder

```bash
# First train a decoder
python chemdiffuse3d/train_decoder.py \
    --task_configs_json configs/3dsr4z_config.json \
    --exp-name 3dsr4z_rcan \
    --decoder-type rcan \
    --save-dir ./outputs/decoder

# Then evaluate with it
CUDA_VISIBLE_DEVICES=0 python chemdiffuse3d/generate.py \
    --task_configs_json "./configs/3dsr4z_config.json" \
    --backbone_args_json "./configs/backbone_config.json" \
    --conditioning_args_json "./configs/encoder_config.json" \
    --ckpt_dir "./outputs/3dsr4z_experiment/checkpoints" \
    --resume-steps 40000 \
    --exp-name "eval_rcan" \
    --decoder-type rcan \
    --decoder-path "./outputs/decoder/rcan/3d_sr/best_decoder.pth" \
    --save-results
```

See `scripts/eval.sh` and `scripts/train_decoder.sh` for more examples.

## Configuration

### Model Configuration

| File | Description |
|------|-------------|
| `configs/backbone_config.json` | SiT-3D backbone hyperparameters (hidden size, depth, heads, etc.) |
| `configs/encoder_config.json` | LR conditioning encoder parameters |
| `configs/3dsr4z_config.json` | 3D super-resolution (4× Z) task setup |
| `configs/3ddenoise_config.json` | 3D denoising task setup |
| `configs/biotisr_config.json` | Isotropic restoration task setup |
| `configs/task_config.json` | Multi-task joint training setup |

### Data Paths

Update the `train_data_dir` and `dev_data_dir` fields in the task config JSON files to point to your HDF5 data files. Replace `<YOUR_DATA_PATH>` with your actual data directory.

## Project Structure

```
ChemDiffuse3D/
├── chemdiffuse3d/               # Main package
│   ├── train.py                 # Multi-task training script
│   ├── generate.py              # Inference / evaluation
│   ├── train_decoder.py         # Post-diffusion decoder training
│   ├── loss.py                  # SILoss (denoising + REPA)
│   ├── samplers.py              # Euler & Euler-Maruyama samplers
│   ├── model/
│   │   ├── model.py             # JointModel wrapper
│   │   ├── sit.py               # Timestep embedder, projectors
│   │   ├── sit_backbone.py      # SiT-3D backbone
│   │   ├── attention.py         # Factorized spatial-depth attention
│   │   ├── unet3d.py            # LR conditioning encoder
│   │   ├── decoder.py           # RCAN & Fused decoders
│   │   ├── utils.py             # Positional embeddings
│   │   └── rope.py              # Rotary position embeddings
│   └── data/
│       ├── dataset.py           # HDF5 dataset classes
│       ├── dataloader.py        # Multi-task data loading
│       └── data_utils.py        # PSF simulation utilities
├── configs/                     # Configuration files
├── scripts/                     # Example training/eval shell scripts
├── data_processing/             # Data preprocessing utilities
└── figures/                     # Figure reproduction scripts
```

## Model Checkpoints

Pre-trained model checkpoints will be made available on [Zenodo/HuggingFace]. *(Link to be added upon publication.)*

## Citation

```bibtex
@article{chemdiffuse3d2026,
  title={ChemDiffuse3D: Conditional 3D Diffusion Model for Volumetric Microscopy Image Restoration},
  author={},
  journal={},
  year={2026}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
