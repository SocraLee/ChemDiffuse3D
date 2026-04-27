"""
Factorized Attention for 3D volumetric data.

Implements spatial-then-depth factorized self-attention using
Diffusers' optimized attention kernel. This decomposition reduces
the quadratic complexity of full 3D attention from O((D*H*W)^2) to
O(D*(HW)^2 + HW*(D)^2).
"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from diffusers.models.attention import Attention as DiffusersAttention


class FactorizedAttentionLayer(nn.Module):
    """
    Factorized self-attention: first across spatial (HW), then across depth (D).

    Given an input of shape (B, D*HW, C), this layer:
    1. Reshapes to (B*D, HW, C) and applies spatial self-attention
    2. Reshapes to (B*HW, D, C) and applies depth self-attention
    3. Returns the result in (B, D*HW, C) shape
    """

    def __init__(self, dim, num_heads, HW_patches, dim_head=None, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.HW_patches = HW_patches
        if dim_head is None:
            dim_head = dim // num_heads

        # Spatial attention (operates on HW_patches tokens, batch = B*D)
        self.spatial_attn = DiffusersAttention(
            query_dim=self.dim,
            cross_attention_dim=None,
            heads=self.num_heads,
            dim_head=dim_head,
        )

        # Depth attention (operates on D tokens, batch = B*HW)
        self.depth_attn = DiffusersAttention(
            query_dim=self.dim,
            cross_attention_dim=None,
            heads=self.num_heads,
            dim_head=dim_head,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (B, T_3D, C), where T_3D = D * HW_patches.
        Returns:
            torch.Tensor: (B, T_3D, C).
        """
        B, T_3D, C = x.shape
        HW = self.HW_patches
        D = int(T_3D // self.HW_patches)
        assert T_3D == D * HW

        # Step 1: Spatial attention over HW
        x_spatial_in = x.contiguous().view(B * D, HW, C)
        x_after_spatial_attn = self.spatial_attn(hidden_states=x_spatial_in)
        x_after_spatial_stage = x_after_spatial_attn.contiguous().view(B, D, HW, C)

        # Step 2: Depth attention over D
        x_depth_in = x_after_spatial_stage.permute(0, 2, 1, 3).contiguous().view(B * HW, D, C)
        x_after_depth_attn = self.depth_attn(hidden_states=x_depth_in)
        x_after_depth_stage = x_after_depth_attn.view(B, HW, D, C)

        # Reshape back to (B, D*HW, C)
        final_output = x_after_depth_stage.permute(0, 2, 1, 3).contiguous().view(B, T_3D, C)
        return final_output
