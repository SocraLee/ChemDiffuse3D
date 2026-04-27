"""
Core SiT (Scalable Interpolant Transformer) components for ChemDiffuse3D.

Includes: TimestepEmbedder, ConditionFuser, REPA projectors (DINOv2, CLIP),
FinalLayer, and dynamic 3D positional embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .utils import get_1d_sincos_pos_embed_from_grid, get_2d_sincos_pos_embed, get_3d_sincos_pos_embed


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations using
    sinusoidal positional encoding followed by an MLP.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.positional_embedding(t, dim=self.frequency_embedding_size).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class ConditionFuser(nn.Module):
    """Fuses DINO embeddings into a global conditioning vector."""

    def __init__(self, emb_dim=1024, inner_dim=512,
                 global_out_dim=1152, dropout_prob=0.0):
        super().__init__()
        self.inner_dim = inner_dim
        self.act = nn.GELU()
        self.emb_proj = nn.Linear(emb_dim, inner_dim)
        self.global_proj = nn.Linear(inner_dim, global_out_dim)

    def forward(self, emb):
        if len(emb.shape) == 3:
            global_cond = torch.mean(self.act(self.emb_proj(emb)), dim=(1))
        elif len(emb.shape) == 4:
            global_cond = torch.mean(self.act(self.emb_proj(emb)), dim=(1, 2))
        global_cond = self.global_proj(global_cond)
        return global_cond


class DinoV2PatchProjector(nn.Module):
    """Projects SiT block outputs to DINOv2 patch embedding space for REPA alignment."""

    def __init__(self, model_latent_dim, projector_intermediate_dim=1024, dinov2_target_dim=4096):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(model_latent_dim, projector_intermediate_dim),
            nn.SiLU(),
            nn.Linear(projector_intermediate_dim, projector_intermediate_dim),
            nn.SiLU(),
            nn.Linear(projector_intermediate_dim, dinov2_target_dim),
        )

    def forward(self, x):
        # x: (B, T, model_latent_dim)
        return self.mlp(x)  # Output: (B, T, dinov2_target_dim)


def modulate(x, shift, scale):
    """Apply adaptive layer norm modulation with clamping for stability."""
    x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return x.clamp(min=-100, max=100)


class FinalLayer(nn.Module):
    """The final layer of SiT with adaptive layer norm."""

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DynamicAbs3DPosEmbed(nn.Module):
    """
    Learnable 3D absolute positional embedding initialized with sin-cos encoding.
    Supports dynamic depth/spatial resolution via trilinear interpolation.
    """

    def __init__(self, channel_dim, max_depth=20, max_hw=16):
        super().__init__()
        self.channel_dim = channel_dim
        self.pos_emb = nn.Parameter(torch.zeros(1, channel_dim, max_depth, max_hw, max_hw), requires_grad=True)
        self.max_depth = max_depth
        self.max_hw = max_hw
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            pos_embed_data = get_3d_sincos_pos_embed(
                self.channel_dim,
                grid_size_d=self.max_depth,
                grid_size_h=self.max_hw,
                grid_size_w=self.max_hw
            )  # (D*H*W, C)
            pos_embed_data = pos_embed_data.view(self.max_depth, self.max_hw, self.max_hw, self.channel_dim)
            pos_embed_data = pos_embed_data.permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, D, H, W)
            self.pos_emb.data.copy_(pos_embed_data)

    def forward(self, x, B, D, H, W):
        x = x.view(B, D, H, W, -1)
        if D != self.pos_emb.shape[2] or (H, W) != (self.pos_emb.shape[3], self.pos_emb.shape[4]):
            pos_emb = F.interpolate(
                self.pos_emb,
                size=(D, W, H),
                mode='trilinear',
                align_corners=False
            )
        else:
            pos_emb = self.pos_emb
        pos_emb = pos_emb.permute(0, 2, 3, 4, 1)  # (1, D, H, W, C)

        x = x + pos_emb
        x = x.contiguous().view(B, D * H * W, self.channel_dim)
        return x
