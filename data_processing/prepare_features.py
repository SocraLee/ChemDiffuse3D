"""
Data Preprocessing: VAE and DINO Feature Extraction.

Processes raw 3D microscopy volumes stored in HDF5 format, extracting:
- VAE latent representations (using Stability AI's sd-vae-ft-mse)
- DINOv2 patch embeddings (using a fine-tuned DINOv2 ViT-L/16)

These precomputed features are stored in the same HDF5 file for
efficient training data loading.

Usage:
    python data_processing/prepare_features.py \
        --data_path <path_to_h5_file> \
        --dino_model_path <path_to_dinov2_checkpoint> \
        --dino_config_path <path_to_dinov2_config>
"""

import argparse
import sys
import os
import h5py
import torch
import numpy as np
from tqdm import tqdm
from diffusers.models import AutoencoderKL


def min_max_normalize(input_tensor: torch.Tensor) -> torch.Tensor:
    """Per-slice min-max normalization."""
    min_val = torch.amin(input_tensor, dim=(-2, -1), keepdim=True)
    max_val = torch.amax(input_tensor, dim=(-2, -1), keepdim=True)
    eps = 1e-5
    normalized_tensor = (input_tensor - min_val) / (max_val - min_val + eps)
    return normalized_tensor


def print_gpu_memory(tag=""):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        tqdm.write(f"[{tag}] GPU Memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
    else:
        tqdm.write("CUDA is not available.")


def load_vae_encoder(device, batch_size=200):
    """Load Stability AI VAE encoder."""
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    return vae


def encode_vae_batch(vae, images, batch_size=200):
    """Encode images with VAE in batches."""
    all_latents = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        # Expand grayscale to 3-channel and scale to [-1, 1]
        vae_input = batch.unsqueeze(1).expand(-1, 3, -1, -1)
        vae_input = vae_input * 2 - 1
        with torch.no_grad():
            dist = vae.encode(vae_input).latent_dist
            latent = torch.cat([dist.mean, dist.std], dim=1)
        all_latents.append(latent)
    return torch.cat(all_latents, dim=0)


def load_dino_model(model_path, config_path, device):
    """
    Load DINOv2 model.

    This function should be adapted to your specific DINOv2 setup.
    The default implementation uses the dinov2 library's setup utilities.

    Args:
        model_path: Path to DINOv2 checkpoint
        config_path: Path to DINOv2 config YAML
        device: Target device
    """
    # Option 1: Using dinov2 library (requires dinov2-main in your path)
    # sys.path.insert(0, '<path_to_dinov2_main>')
    # from dinov2.eval.setup import setup_and_build_model_eval, get_args_parser
    # args = get_args_parser()
    # args.config_file = config_path
    # args.pretrained_weights = model_path
    # args.skip_distributed = True
    # model, _ = setup_and_build_model_eval(args)

    # Option 2: Using HuggingFace Transformers
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        "facebook/dinov2-large",
        trust_remote_code=True
    ).to(device)

    model.eval()
    return model


def extract_dino_features(dino_model, images, device):
    """Extract DINOv2 patch token features from a batch of images."""
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    dino_input = images.unsqueeze(1).expand(-1, 3, -1, -1)
    dino_input = (dino_input - IMAGENET_MEAN) / IMAGENET_STD

    with torch.no_grad():
        features = dino_model.forward_features(dino_input)['x_norm_patchtokens']
    return features


def process_h5(data_dir, task_configs, dino_model_path=None, dino_config_path=None):
    """
    Process an HDF5 file, adding VAE latents and DINO features.

    Args:
        data_dir: Path to the HDF5 file
        task_configs: Dict mapping task_key -> {batch_size, chunk_size}
        dino_model_path: Path to DINOv2 checkpoint
        dino_config_path: Path to DINOv2 config
    """
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load models
    vae = load_vae_encoder(DEVICE)
    dino_model = load_dino_model(dino_model_path, dino_config_path, DEVICE)

    with h5py.File(data_dir, 'a') as f:
        for key, task_config in task_configs.items():
            suffix = key.lower()
            data = f[key]
            batch_size = task_config['batch_size']

            HR_CUBE_KEY = f"hr_cube_{suffix}"
            VAE_HR_OUT_KEY = f"vae_hr_cube_{suffix}"
            DINO_HR_OUT_KEY = f"dino_hr_cube_{suffix}"

            num_cubes = data.shape[0]
            hr_depth, hr_h, hr_w = data.shape[1:]

            datasets_to_create = {
                HR_CUBE_KEY: ((num_cubes, hr_depth, hr_h, hr_w), np.float32),
                VAE_HR_OUT_KEY: ((num_cubes, hr_depth, 8, 32, 32), np.float32),
                DINO_HR_OUT_KEY: ((num_cubes, hr_depth, 256, 1024), np.float32),
            }

            for ds_name, (ds_shape, ds_type) in datasets_to_create.items():
                if ds_name in f:
                    del f[ds_name]
                f.create_dataset(
                    ds_name, shape=ds_shape, dtype=ds_type,
                    chunks=(task_config['chunk_size'],) + ds_shape[1:],
                    compression='lzf'
                )

            for idx in tqdm(range(0, num_cubes, batch_size), desc=f"Processing {key}"):
                batch = torch.from_numpy(data[idx:idx + batch_size]).float().to(DEVICE)
                batch = min_max_normalize(batch)

                f[HR_CUBE_KEY][idx:idx + batch_size] = batch.cpu().numpy()
                Bh, Dh, H, W = batch.shape
                hr_tensor = batch.reshape(Bh * Dh, H, W)

                # VAE encoding
                vae_latent = encode_vae_batch(vae, hr_tensor)
                vae_latent = vae_latent.reshape(Bh, Dh, 8, 32, 32).cpu().numpy()

                # DINO feature extraction
                dino_features = extract_dino_features(dino_model, hr_tensor, DEVICE)
                dino_features = dino_features.reshape(Bh, Dh, 256, 1024).cpu().numpy()

                f[VAE_HR_OUT_KEY][idx:idx + batch_size] = vae_latent
                f[DINO_HR_OUT_KEY][idx:idx + batch_size] = dino_features

    print("Processing complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess microscopy data for MicroDiffuse3D")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the HDF5 file containing raw microscopy volumes")
    parser.add_argument("--dino_model_path", type=str, default=None,
                        help="Path to DINOv2 checkpoint (optional, uses HuggingFace default if not provided)")
    parser.add_argument("--dino_config_path", type=str, default=None,
                        help="Path to DINOv2 config YAML (optional)")

    args = parser.parse_args()

    # Example task configurations - adjust batch_size and chunk_size
    # based on your GPU memory and dataset size
    task_configs = {
        'Spatial_01': {"batch_size": 600, "chunk_size": 200},
        'Spatial_10': {"batch_size": 60, "chunk_size": 10},
        'Spatial_20': {"batch_size": 30, "chunk_size": 5},
        'Spectral_01': {"batch_size": 600, "chunk_size": 200},
        'Spectral_10': {"batch_size": 60, "chunk_size": 10},
        'Spectral_20': {"batch_size": 30, "chunk_size": 5},
    }

    process_h5(args.data_path, task_configs,
               dino_model_path=args.dino_model_path,
               dino_config_path=args.dino_config_path)
