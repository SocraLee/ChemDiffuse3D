"""
Utility functions for positional embeddings.

Provides sinusoidal positional encoding for 1D, 2D, and 3D grids,
used in both the SiT backbone and the factorized positional embeddings.
"""

import torch
import numpy as np
import math
import logging
from typing import List, Tuple
from torch import Tensor
from torchvision.utils import make_grid


def array2grid(x):
    """Convert a batch of images to a grid for visualization."""
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    """Sample from the VAE posterior distribution."""
    device = moments.device
    mean, std = torch.chunk(moments, 2, dim=-3)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias)
    return z


def create_logger(logging_dir):
    """Create a logger that writes to a log file and stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """Set requires_grad flag for all parameters in a model."""
    for p in model.parameters():
        p.requires_grad = flag


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Generate 1D sinusoidal positional embedding.

    Args:
        embed_dim: output dimension for each position
        pos: a tensor of positions to be encoded, shape (M,)
    Returns:
        Tensor of shape (M, embed_dim)
    """
    assert embed_dim % 2 == 0, f"{embed_dim} is not even"
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.
    omega = 1. / (10000 ** omega)

    pos = pos.reshape(-1)
    out = pos[:, None] * omega[None, :]

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w):
    """
    Generate 2D sinusoidal positional embedding.

    Returns:
        pos_embed: [grid_size_h * grid_size_w, embed_dim]
    """
    grid_h = torch.arange(grid_size_h, dtype=torch.float32)
    grid_w = torch.arange(grid_size_w, dtype=torch.float32)
    mesh_h, mesh_w = torch.meshgrid(grid_h, grid_w, indexing='ij')
    mesh_h = mesh_h.flatten()
    mesh_w = mesh_w.flatten()

    total_pairs = embed_dim // 2
    pairs_h = total_pairs // 2
    pairs_w = total_pairs - pairs_h
    dim_h = pairs_h * 2
    dim_w = pairs_w * 2

    emb_h = get_1d_sincos_pos_embed_from_grid(dim_h, mesh_h)
    emb_w = get_1d_sincos_pos_embed_from_grid(dim_w, mesh_w)

    pos_embed = torch.cat([emb_h, emb_w], dim=1)
    return pos_embed


def get_3d_sincos_pos_embed(embed_dim, grid_size_d, grid_size_h, grid_size_w,
                            dtype=torch.float32):
    """
    Generate 3D sinusoidal positional embedding.

    Returns:
        pos_embed: [grid_size_d * grid_size_h * grid_size_w, embed_dim]
    """
    grid_d = torch.arange(grid_size_d, dtype=torch.float32)
    grid_h = torch.arange(grid_size_h, dtype=torch.float32)
    grid_w = torch.arange(grid_size_w, dtype=torch.float32)

    # Divide cos/sin pairs evenly across D, H, W dimensions
    total_pairs = embed_dim // 2
    pairs_d = total_pairs // 3
    pairs_h = total_pairs // 3
    pairs_w = total_pairs - pairs_d - pairs_h
    dim_d = pairs_d * 2
    dim_h = pairs_h * 2
    dim_w = pairs_w * 2

    emb_d = get_1d_sincos_pos_embed_from_grid(dim_d, grid_d)
    emb_h = get_1d_sincos_pos_embed_from_grid(dim_h, grid_h)
    emb_w = get_1d_sincos_pos_embed_from_grid(dim_w, grid_w)

    pos_embed = torch.cat([
        emb_d.unsqueeze(1).unsqueeze(1).repeat(1, grid_size_h, grid_size_w, 1),
        emb_h.unsqueeze(0).unsqueeze(2).repeat(grid_size_d, 1, grid_size_w, 1),
        emb_w.unsqueeze(0).unsqueeze(0).repeat(grid_size_d, grid_size_h, 1, 1)
    ], dim=3).reshape(grid_size_d * grid_size_h * grid_size_w, embed_dim)

    return pos_embed.to(dtype)
