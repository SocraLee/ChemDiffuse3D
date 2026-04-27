"""
Dataset classes for ChemDiffuse3D.

Provides HDF5-based datasets for loading preprocessed 3D microscopy
volumes along with their VAE latents and DINO embeddings.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
import torch.nn as nn
from torchvision.transforms import v2
from .data_utils import RandomMicroscopeDegradation


def get_file_index(file_path):
    basename = os.path.basename(file_path)
    index_part = basename.split('_')[-1].split('.')[0]
    return int(index_part)


class H5Dataset(Dataset):
    """
    HDF5 dataset for loading preprocessed 3D microscopy cubes.

    Each sample contains:
    - hr: High-resolution volume (D, 1, H, W)
    - dino_lr: DINO embeddings of LR slices (D, T, C)
    - vae_hr: VAE latent of HR slices (D, 8, 32, 32)
    - lr: Low-resolution volume (1, D, H, W)
    - pretrained_embedding: Optional REPA alignment targets
    """

    def __init__(self, h5_path, z_types=None, if_train=True):
        if z_types is None:
            z_types = {}
        self.h5_path = h5_path
        self.if_train = if_train
        self.h5_file = None  # Opened lazily in worker processes
        self.z_types = z_types

        with h5py.File(self.h5_path, 'r') as temp_f:
            if "hr_denoise_cube" in temp_f:
                self.hr_key = "hr_denoise_cube"
            else:
                self.hr_key = "hr_cube"
            self.total_samples = temp_f[self.hr_key].shape[0]
            self.lr_key = "lr_cube"

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        dino_lr = torch.from_numpy(self.h5_file['dino_lr_cube'][idx]).float()
        vae_hr = torch.from_numpy(self.h5_file['vae_hr_cube'][idx]).float()
        hr = torch.from_numpy(self.h5_file[self.hr_key][idx]).float().unsqueeze(1)
        lr = torch.from_numpy(self.h5_file[self.lr_key][idx]).float().unsqueeze(0)

        if not self.if_train:
            return hr, dino_lr, vae_hr, 0, lr
        else:
            pretrained_embedding = {}
            if "dinov2" in self.z_types:
                pretrained_embedding["dinov2"] = torch.from_numpy(
                    self.h5_file['dino_hr_cube'][idx]).float()
            return hr, dino_lr, vae_hr, pretrained_embedding, lr


def h5_worker_init_fn(worker_id):
    """Initialize HDF5 file handle in each DataLoader worker process."""
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.h5_file = h5py.File(dataset.h5_path, 'r')


def min_max_normalize(input_tensor: torch.Tensor) -> torch.Tensor:
    min_val = torch.amin(input_tensor, dim=(-2, -1), keepdim=True)
    max_val = torch.amax(input_tensor, dim=(-2, -1), keepdim=True)
    eps = 1e-5
    normalized_tensor = (input_tensor - min_val) / (max_val - min_val + eps)
    return normalized_tensor


def interpolate_single_stack(lr, desired_depth):
    assert len(lr.shape) == 3
    D, _, _ = lr.shape
    if D != desired_depth:
        lr_image_5d = lr.unsqueeze(0).unsqueeze(0)
        lr_image_interp = torch.nn.functional.interpolate(
            lr_image_5d,
            size=(desired_depth, lr.shape[-2], lr.shape[-1]),
            mode='trilinear',
            align_corners=False
        )
        lr_image_interp = lr_image_interp.squeeze(0).squeeze(0)
        lr_image_interp = min_max_normalize(lr_image_interp)
    else:
        lr_image_interp = lr
    return lr_image_interp


# Augmentation pipelines for self-supervised pretraining
spectral_pipeline = v2.Compose([
    v2.RandomErasing(p=0.9, scale=(0.5, 0.75)),
])

spatial_pipeline_corruption = v2.Compose([
    v2.RandomErasing(p=0.9, scale=(0.5, 0.75)),
])

spatial_pipeline_degradation = RandomMicroscopeDegradation()


class KeyBasedPretrainDataset(Dataset):
    """
    Dataset for self-supervised pretraining with on-the-fly degradation.

    Supports both spatial degradation (PSF convolution + downsampling) and
    spectral degradation (random erasing) modes.
    """

    def __init__(self, h5_path, z_types=None, key=""):
        if z_types is None:
            z_types = {}
        self.h5_path = h5_path
        self.h5_file = None
        self.z_types = z_types
        self.key = key
        self.valid_key = f"valid_{self.key}"
        self.use_valid_key = True

        with h5py.File(self.h5_path, 'r') as temp_f:
            if self.valid_key in temp_f:
                self.total_samples = temp_f[self.valid_key].shape[0]
            else:
                k = f"hr_cube_{self.key.lower()}"
                assert k in temp_f, f"{k} not in {temp_f.keys()}"
                self.total_samples = len(temp_f[k])
                self.use_valid_key = False

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
        if self.use_valid_key:
            idx = self.h5_file[self.valid_key][idx]
        vae_hr = torch.from_numpy(self.h5_file[f'vae_hr_cube_{self.key.lower()}'][idx]).float()
        hr = torch.from_numpy(self.h5_file[f'hr_cube_{self.key.lower()}'][idx]).float()
        D, W, H = hr.shape

        if "Spatial" in self.key:
            if D != 1:
                lr, _ = spatial_pipeline_degradation(hr)
                lr = interpolate_single_stack(lr, hr.shape[0])
            else:
                lr = spatial_pipeline_corruption(hr)
        else:
            lr = spectral_pipeline(hr)

        hr = hr.unsqueeze(1)  # (D, 1, W, H) for per-slice metrics
        lr = lr.unsqueeze(0)  # (1, D, W, H) for 3D convolution

        pretrained_embedding = {}
        if "dinov2" in self.z_types:
            pretrained_embedding["dinov2"] = torch.from_numpy(
                self.h5_file[f'dino_hr_cube_{self.key.lower()}'][idx]).float()

        return hr, 0, vae_hr, pretrained_embedding, lr
