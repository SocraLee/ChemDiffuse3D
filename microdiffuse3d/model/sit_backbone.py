"""
SiT-3D Backbone for MicroDiffuse3D.

Implements the 3D Scalable Interpolant Transformer backbone with
factorized spatial-depth attention and REPA (REPresentation Alignment)
projection heads.
"""

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp
from .attention import FactorizedAttentionLayer
from .sit import DinoV2PatchProjector, modulate, FinalLayer, DynamicAbs3DPosEmbed


class SiTBlock3D(nn.Module):
    """
    A 3D SiT block using FactorizedAttentionLayer for spatial-depth
    factorized self-attention with AdaLN modulation.
    """

    def __init__(self, hidden_size, num_heads, HW_patches, cond_dim, mlp_ratio=4.0, dim_head=None, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = FactorizedAttentionLayer(
            dim=hidden_size, num_heads=num_heads, dim_head=dim_head, HW_patches=HW_patches,
            is_cross_attention=False
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        self.norm3 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.cond_projector = nn.Linear(cond_dim, hidden_size)
        self.act = nn.GELU()

    def forward(self, x, c, condition):
        (shift_msa, scale_msa, gate_msa,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c).chunk(6, dim=-1)

        condition = self.act(self.cond_projector(condition))
        x = x + self.norm3(condition)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class SiT_3D_Backbone(nn.Module):
    """
    3D Scalable Interpolant Transformer Backbone.

    Processes volumetric VAE latents with factorized spatial-depth attention,
    conditioned on LR image embeddings via concatenation or addition.
    Supports optional REPA projection heads for representation alignment.
    """

    def __init__(
            self,
            input_size=32,
            encoder_depth=8,
            patch_size=2,
            in_channels=4,
            model_depth=28,
            hidden_size=1152,
            lr_embedding_dim=256,
            num_heads=16,
            dim_head=32,
            mlp_ratio=4.0,
            z_types=[],
            condition_dim=2048,
            dino_feat_dim=1024,
            cond_inject_method="concat",
            **block_kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.depth_size = model_depth
        self.encoder_depth = encoder_depth
        self.hidden_size = hidden_size
        self.cond_inject_method = cond_inject_method

        if self.cond_inject_method == "concat":
            self.concated_hidden = lr_embedding_dim + hidden_size
        elif self.cond_inject_method == "add":
            self.concated_hidden = hidden_size
        else:
            raise ValueError(f"Unknown cond_inject_method: {self.cond_inject_method}")

        self.z_types = z_types

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.HW_patches = self.x_embedder.num_patches

        self.pos_emb = DynamicAbs3DPosEmbed(
            channel_dim=self.concated_hidden,
            max_hw=int(self.HW_patches ** 0.5)
        )

        self.blocks = nn.ModuleList([
            SiTBlock3D(
                self.concated_hidden, num_heads,
                dim_head=dim_head,
                HW_patches=self.HW_patches,
                cond_dim=condition_dim,
                mlp_ratio=mlp_ratio,
                **block_kwargs
            ) for _ in range(model_depth)
        ])

        # REPA projection heads
        self.projectors = nn.ModuleDict()
        for z_type in self.z_types:
            if z_type == 'dinov2':
                self.projectors[z_type] = DinoV2PatchProjector(
                    model_latent_dim=self.concated_hidden, dinov2_target_dim=dino_feat_dim)
            else:
                raise ValueError(f'Unknown z_type: {z_type}')

        self.final_layer = FinalLayer(self.concated_hidden, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.GroupNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify_3d(self, x):
        """
        Unpatchify 3D token sequence back to volumetric output.

        Args:
            x: (B, D*T_spatial, P*P*C_out)
        Returns:
            imgs: (B, D, C_out, H, W)
        """
        B, T_3D, C_in = x.shape

        c = self.out_channels
        p = self.patch_size
        h_patch = w_patch = int(self.HW_patches ** 0.5)
        d_patch = int(T_3D // self.HW_patches)

        x_flat = x.view(B * d_patch, self.HW_patches, p * p * c)
        x_flat = x_flat.reshape(shape=(x_flat.shape[0], h_patch, w_patch, p, p, c))
        x_flat = torch.einsum('nhwpqc->nchpwq', x_flat)
        imgs_flat = x_flat.reshape(shape=(x_flat.shape[0], c, h_patch * p, w_patch * p))

        imgs = imgs_flat.view(B, d_patch, c, h_patch * p, w_patch * p)
        return imgs

    def forward(self, x, c_global, c_concat, c_cross_attn, feature_alignment=False):
        """
        Forward pass of the SiT-3D backbone.

        Args:
            x: (B, D, C_in, H, W) - 3D VAE Latent
            c_global: (B, C_global) - Global condition for AdaLN (timestep + DINO embedding)
            c_concat: (B, T_3D, C_concat) - LR conditioning (concat branch)
            c_cross_attn: (B, T_3D, C_cross) - LR conditioning (additive branch)
            feature_alignment: Whether to compute REPA projections at encoder_depth
        """
        B, D, C_in, H, W = x.shape
        x_flat = x.view(B * D, C_in, H, W)
        x_tokens_flat = self.x_embedder(x_flat)
        x_3d_tokens = x_tokens_flat.contiguous().view(
            B, D * self.HW_patches, self.hidden_size
        )

        if self.cond_inject_method == "concat":
            x_3d_tokens = torch.cat([x_3d_tokens, c_concat], dim=-1)
        elif self.cond_inject_method == "add":
            x_3d_tokens = x_3d_tokens + c_concat
        else:
            raise ValueError(f"Unknown cond_inject_method: {self.cond_inject_method}")

        x_3d_tokens = x_3d_tokens + self.pos_emb(
            x_3d_tokens, B, D,
            int(self.HW_patches ** 0.5),
            int(self.HW_patches ** 0.5)
        )

        # Transformer blocks with optional REPA projection
        zs = None
        for i, block in enumerate(self.blocks):
            x_3d_tokens = block(x_3d_tokens, c_global, c_cross_attn)
            if feature_alignment and (i + 1) == self.encoder_depth:
                zs = {}
                x_for_proj = x_3d_tokens
                for z_type in self.projectors:
                    z_flat = self.projectors[z_type](x_for_proj)
                    zs[z_type] = z_flat

        # Final layer
        x_final_tokens = self.final_layer(x_3d_tokens, c_global)
        x_out = self.unpatchify_3d(x_final_tokens)

        return x_out, zs
