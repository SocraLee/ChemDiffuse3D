"""
JointModel: Top-level model wrapper for ChemDiffuse3D.

Composes the SiT-3D backbone, timestep embedder, condition fuser,
and LR conditioning module into a single nn.Module.
"""

import torch.nn as nn
from .sit_backbone import SiT_3D_Backbone
from .sit import TimestepEmbedder, ConditionFuser
from .unet3d import Shared_LR_Conditioning_Module


class JointModel(nn.Module):
    def __init__(self, backbone_args, conditioning_args, common_z_types):
        """
        Args:
            backbone_args (dict): Configuration for SiT_3D_Backbone. Example:
                {
                    "condition_dim": 2048,
                    "lr_embedding_dim": 256,
                }
            conditioning_args (dict): Configuration for LR conditioning encoder. Example:
                {
                    "lr_encoder_params": {
                        "out_embed_dim": 768,
                        "out_concat_dim": 256,
                        "norm_groups": 32,
                        "dropout_p": 0.0
                    }
                }
            common_z_types (list): Types for representation alignment projectors,
                e.g., ['dinov2']
        """
        super().__init__()
        self.cond_inject_method = backbone_args.get("cond_inject_method", "concat")
        if self.cond_inject_method == "add":
            latent_dim = backbone_args.get("hidden_size", 1152)
            backbone_args["lr_embedding_dim"] = latent_dim
            if "lr_encoder_params" in conditioning_args:
                conditioning_args["lr_encoder_params"]["out_concat_dim"] = latent_dim

        self.backbone = SiT_3D_Backbone(z_types=common_z_types, **backbone_args)
        shared_concated_hidden = self.backbone.concated_hidden
        self.t_embedder = TimestepEmbedder(shared_concated_hidden)
        self.lr_condition_modules = Shared_LR_Conditioning_Module(conditioning_args=conditioning_args)
        self.y_embedders = ConditionFuser(global_out_dim=shared_concated_hidden)

    def forward(self, x_3d, t, y, lr_image, feature_alignment=False):
        """
        Args:
            x_3d (torch.Tensor): (B, D_hr, C, H, W) - VAE Latent
            t (torch.Tensor): (B,) - Timesteps
            y (torch.Tensor): (B, D_hr, T_spatial, C_dino) - 3D DINO embeddings
            lr_image (torch.Tensor): (B, C_lr, D_lr, H_lr, W_lr) - 3D LR volume
            feature_alignment (bool): Whether to compute REPA projection outputs
        """
        c_additive_raw, c_concat = self.lr_condition_modules(lr_image)
        t_emb = self.t_embedder(t)

        if len(y.shape) == 4:
            global_cond = self.y_embedders(y)
            c_global = t_emb + global_cond
        else:
            c_global = t_emb

        x_out, zs = self.backbone(
            x_3d,
            c_global,
            c_concat,
            c_additive_raw,
            feature_alignment=feature_alignment
        )

        return x_out, zs
