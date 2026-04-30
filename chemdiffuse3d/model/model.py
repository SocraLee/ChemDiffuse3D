"""
JointModel: Top-level model wrapper for ChemDiffuse3D.

Composes the SiT-3D backbone, timestep embedder, condition fuser,
and LR conditioning module into a single nn.Module.
"""

import torch.nn as nn
from .sit_backbone import SiT_3D_Backbone
from .sit import TimestepEmbedder, ConditionFuser
from .unet3d import Shared_LR_Conditioning_Module


def _resolve_decoder_flags(task_configs):
    """
    Decide which conditioning-decoder branches to build from task_configs.

    Per-task rule (only triggers if both 'input_depth' and 'output_depth' are present):
      - input_depth == output_depth          -> needs the matched-depth decoder
      - output_depth == 4 * input_depth      -> needs the depth upsampler
      - other ratios                         -> ignored (caller is expected to
        pre-interpolate to matched depth and use the decoder branch)
      - missing fields                       -> contributes nothing

    If task_configs is None, defaults to (True, True) for backward compatibility.
    """
    if task_configs is None:
        return True, True
    with_upsampler = False
    with_decoder = False
    for cfg in task_configs.values():
        i = cfg.get('input_depth')
        o = cfg.get('output_depth')
        if i is None or o is None:
            continue
        if i == o:
            with_decoder = True
        elif o == 4 * i:
            with_upsampler = True
        else:
            raise ValueError(f"Unsupported depth ratio: {o} / {i}.\nYou can use pre-interpolation or add your own upsampler/decoder branch in Shared_LR_Conditioning_Module.")
    return with_upsampler, with_decoder


class JointModel(nn.Module):
    def __init__(self, backbone_args, conditioning_args, common_z_types, task_configs=None):
        """
        Args:
            backbone_args (dict): Configuration for SiT_3D_Backbone.
            conditioning_args (dict): Configuration for LR conditioning encoder.
            common_z_types (list): Types for representation alignment projectors,
                e.g., ['dinov2']
            task_configs (dict | None): Optional task_configs; used to decide
                whether to build the depth-upsampler and/or matched-depth decoder
                inside the conditioning module. If None, both branches are built.
        """
        super().__init__()
        self.cond_inject_method = backbone_args.get("cond_inject_method", "concat")
        if self.cond_inject_method == "add":
            latent_dim = backbone_args.get("hidden_size", 1152)
            backbone_args["lr_embedding_dim"] = latent_dim
            if "lr_encoder_params" in conditioning_args:
                conditioning_args["lr_encoder_params"]["out_concat_dim"] = latent_dim

        with_upsampler, with_decoder = _resolve_decoder_flags(task_configs)

        self.backbone = SiT_3D_Backbone(z_types=common_z_types, **backbone_args)
        shared_concated_hidden = self.backbone.concated_hidden
        self.t_embedder = TimestepEmbedder(shared_concated_hidden)
        self.lr_condition_modules = Shared_LR_Conditioning_Module(
            conditioning_args=conditioning_args,
            with_upsampler=with_upsampler,
            with_decoder=with_decoder,
        )
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
