"""
3D UNet-based LR Conditioning Module for MicroDiffuse3D.

Encodes the low-resolution 3D microscopy volume into conditioning
signals for the SiT-3D backbone: a dense embedding for cross-attention
and a compressed embedding for token concatenation.
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
from .attention import FactorizedAttentionLayer


class FeatureWiseAffine3D(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super().__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample3D(nn.Module):
    def __init__(self, dim, dim_out=None, scale_factor=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.conv = nn.Conv3d(dim, dim_out if dim_out is not None else dim, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample3D(nn.Module):
    def __init__(self, dim, dim_out=None, stride=2):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim_out if dim_out is not None else dim, kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        return self.conv(x)


class Block3D(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv3d(dim, dim_out, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock3D(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0.0, use_affine_level=False, norm_groups=32):
        super().__init__()
        if noise_level_emb_dim is not None:
            self.noise_func = FeatureWiseAffine3D(noise_level_emb_dim, dim_out, use_affine_level)
        else:
            self.noise_func = None

        self.block1 = Block3D(dim, dim_out, groups=norm_groups)
        self.block2 = Block3D(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv3d(dim, dim_out, kernel_size=1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)
        if self.noise_func is not None and time_emb is not None:
            h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class FactorizedSelfAttention3D(nn.Module):
    """3D self-attention using factorized spatial-depth decomposition."""

    def __init__(self, in_channel, n_head, H_feat, W_feat, norm_groups=32):
        super().__init__()
        self.H_feat = H_feat
        self.W_feat = W_feat

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.attn_layer = FactorizedAttentionLayer(
            dim=in_channel,
            num_heads=n_head,
            HW_patches=self.H_feat * self.W_feat,
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        assert H == self.H_feat and W == self.W_feat, \
            f"Input shape ({D}, {H}, {W}) doesn't match expected (D, {self.H_feat}, {self.W_feat})"

        residual = x
        x_norm = self.norm(x)

        # Reshape: (B, C, D, H, W) -> (B, D*H*W, C)
        x_tokens = x_norm.view(B, C, D * H * W).transpose(1, 2)
        attn_out_tokens = self.attn_layer(x=x_tokens)

        # Reshape back: (B, D*H*W, C) -> (B, C, D, H, W)
        attn_out_reshaped = attn_out_tokens.transpose(1, 2).contiguous().view(B, C, D, H, W)

        return attn_out_reshaped + residual


class ResnetBlockWithAttn3D(nn.Module):
    def __init__(self, dim, dim_out, *, D_feat, H_feat, W_feat, n_head=10,
                 noise_level_emb_dim=None, norm_groups=32, dropout=0.0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock3D(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = FactorizedSelfAttention3D(dim_out, n_head=n_head,
                                                  H_feat=H_feat, W_feat=W_feat,
                                                  norm_groups=norm_groups)

    def forward(self, x, time_emb=None):
        x = self.res_block(x, time_emb)
        if self.with_attn:
            x = self.attn(x)
        return x


class EncoderModule(nn.Module):
    """
    3D UNet encoder that processes LR volumes from full resolution
    down to a compact bottleneck representation.
    """

    def __init__(
            self,
            lr_input_depth=5,
            lr_input_channels=1,
            lr_input_res_hw=256,
            stem_base_channels=32,
            stem_kernel_size=3,
            stem_stride=(1, 2, 2),
            stem_num_resnet_blocks=1,
            stem_downsample_stages=2,
            core_base_channels_init_factor=2,
            core_channel_mults=(1, 2, 4),
            core_attn_res_divisors=[(1, 4, 4)],
            core_res_blocks_per_stage=2,
            core_downsample_stride=(1, 2, 2),
            mid_res_blocks=2,
            norm_groups=8,
            dropout_p=0.0,
            n_attn_heads=8
    ):
        super().__init__()
        # --- Stem ---
        stem_layers = []
        current_channels = lr_input_channels
        stem_layers.append(nn.Conv3d(current_channels, stem_base_channels, kernel_size=stem_kernel_size, stride=1,
                                     padding=stem_kernel_size // 2))
        current_channels = stem_base_channels
        current_depth, current_hw_res = lr_input_depth, lr_input_res_hw
        for _ in range(stem_num_resnet_blocks):
            stem_layers.append(
                ResnetBlock3D(current_channels, current_channels, norm_groups=norm_groups, dropout=dropout_p))
        for i in range(stem_downsample_stages):
            dim_out_stem_ds = current_channels * 2
            stem_layers.append(
                ResnetBlock3D(current_channels, dim_out_stem_ds, norm_groups=norm_groups, dropout=dropout_p))
            stem_layers.append(Downsample3D(dim_out_stem_ds, stride=stem_stride))
            current_channels = dim_out_stem_ds
            current_depth = math.ceil(current_depth / stem_stride[0])
            current_hw_res = math.ceil(current_hw_res / stem_stride[1])
        self.stem = nn.Sequential(*stem_layers)
        stem_output_channels = current_channels
        stem_output_res_d, stem_output_res_hw = current_depth, current_hw_res

        # --- Core Downsampling Path ---
        downs = []
        core_base_channels = stem_output_channels * core_base_channels_init_factor
        pre_channel = stem_output_channels
        now_res_d, now_res_hw = stem_output_res_d, stem_output_res_hw
        actual_core_attn_res_tuples = []
        for div_tuple in core_attn_res_divisors:
            actual_core_attn_res_tuples.append((math.ceil(stem_output_res_d / div_tuple[0]),
                                                math.ceil(stem_output_res_hw / div_tuple[1]),
                                                math.ceil(stem_output_res_hw / div_tuple[2])))
        for i, mult in enumerate(core_channel_mults):
            dim_out = int(core_base_channels * mult)
            for _ in range(core_res_blocks_per_stage):
                do_attn = (now_res_d, now_res_hw, now_res_hw) in actual_core_attn_res_tuples
                downs.append(
                    ResnetBlockWithAttn3D(dim=pre_channel, dim_out=dim_out, D_feat=now_res_d, H_feat=now_res_hw,
                                          W_feat=now_res_hw, norm_groups=norm_groups, dropout=dropout_p,
                                          with_attn=do_attn, n_head=n_attn_heads))
                pre_channel = dim_out
            if i < len(core_channel_mults) - 1:
                downs.append(Downsample3D(pre_channel, stride=core_downsample_stride))
                now_res_d = math.ceil(now_res_d / core_downsample_stride[0])
                now_res_hw = math.ceil(now_res_hw / core_downsample_stride[1])
        self.downs = nn.ModuleList(downs)
        self.bottleneck_channels = pre_channel
        self.bottleneck_res_d, self.bottleneck_res_hw = now_res_d, now_res_hw

        # --- Mid Block ---
        mid_layers = []
        for _ in range(mid_res_blocks):
            mid_layers.append(
                ResnetBlockWithAttn3D(self.bottleneck_channels, self.bottleneck_channels, D_feat=self.bottleneck_res_d,
                                      H_feat=self.bottleneck_res_hw, W_feat=self.bottleneck_res_hw,
                                      norm_groups=norm_groups, dropout=dropout_p, with_attn=True, n_head=n_attn_heads))
        self.mid = nn.ModuleList(mid_layers)

    def forward(self, x):
        x = self.stem(x)
        for layer in self.downs:
            x = layer(x)
        for layer in self.mid:
            x = layer(x)
        return x


class DecoderModule(nn.Module):
    """
    Decoder that produces both dense and compressed conditioning embeddings
    from the encoder bottleneck features.

    Either or both of the depth-upsampler (D=5 -> D=20, for 4x SR tasks) and the
    matched-depth decoder branches can be built, controlled by `with_upsampler`
    and `with_decoder`. The branch chosen at forward time depends on the input
    depth.
    """

    def __init__(self, in_channels,
                 out_embedding_dim, out_concat_dim,
                 norm_groups=32, dropout_p=0.0,
                 with_upsampler=True, with_decoder=True):
        super().__init__()

        if with_upsampler:
            channels = [in_channels]
            current_channels = in_channels
            num_upsamples = 2
            upsampler_layers = []
            for _ in range(num_upsamples - 1):
                if current_channels // 2 >= out_embedding_dim:
                    current_channels //= 2
                channels.append(current_channels)
            channels.append(out_embedding_dim)
            for i in range(num_upsamples):
                in_c = channels[i]
                out_c = channels[i + 1]
                block = nn.Sequential(
                    ResnetBlock3D(in_c, in_c, norm_groups=norm_groups, dropout=dropout_p),
                    nn.ConvTranspose3d(in_c, out_c, kernel_size=(4, 1, 1), stride=(2, 1, 1),
                                       padding=(1, 0, 0)),
                    Swish()
                )
                upsampler_layers.append(block)
            self.upsampler = nn.Sequential(*upsampler_layers)

        if with_decoder:
            decoder_layers = [
                nn.Sequential(
                    ResnetBlock3D(in_channels, in_channels, norm_groups=norm_groups, dropout=dropout_p),
                    Swish(),
                    ResnetBlock3D(in_channels, out_embedding_dim, norm_groups=norm_groups, dropout=dropout_p),
                    Swish()
                )
            ]
            self.decoder = nn.Sequential(*decoder_layers)

        self.compression_head = nn.Conv3d(out_embedding_dim, out_concat_dim, kernel_size=1)

    def forward(self, x):
        b, c, d, h, w = x.shape
        if d == 5:
            if not hasattr(self, 'upsampler'):
                raise RuntimeError(
                    f"Got LR input depth={d} but upsampler was not built. "
                    "Add a task with output_depth/input_depth == 4 to task_configs."
                )
            x = self.upsampler(x)
        else:
            if not hasattr(self, 'decoder'):
                raise RuntimeError(
                    f"Got LR input depth={d} but matched-depth decoder was not built. "
                    "Add a task with input_depth == output_depth to task_configs."
                )
            x = self.decoder(x)
        compressed_x = self.compression_head(x)
        return x, compressed_x


class Shared_LR_Conditioning_Module(nn.Module):
    """
    Shared LR conditioning module that processes 3D low-resolution volumes
    into conditioning signals for the diffusion backbone.

    Produces two outputs:
    - embedding: Dense feature embedding (B, Seq, out_embed_dim) for cross-attention
    - compressed_embedding: Compressed embedding (B, Seq, out_concat_dim) for concatenation
    """

    def __init__(self, conditioning_args, with_upsampler=True, with_decoder=True):
        super().__init__()
        params = conditioning_args['lr_encoder_params']
        self.encoder = EncoderModule(
            norm_groups=params.get('norm_groups', 8),
            dropout_p=params.get('dropout_p', 0.0)
        )
        self.decoder = DecoderModule(
            in_channels=self.encoder.bottleneck_channels,
            out_embedding_dim=params.get('out_embed_dim'),
            out_concat_dim=params.get('out_concat_dim'),
            norm_groups=params.get('norm_groups', 8),
            dropout_p=params.get('dropout_p', 0.0),
            with_upsampler=with_upsampler,
            with_decoder=with_decoder,
        )

    def forward(self, lr_image_3d):
        if lr_image_3d.ndim == 4:
            x = lr_image_3d.unsqueeze(1)
        elif lr_image_3d.ndim == 5:
            x = lr_image_3d
        else:
            raise ValueError(f"Unexpected lr_image_3d ndim: {lr_image_3d.ndim}")

        feat = self.encoder(x)
        embedding, compressed_embedding = self.decoder(feat)

        b, c1, d, h, w = compressed_embedding.shape
        compressed_embedding = compressed_embedding.reshape(b, c1, d * h * w).permute(0, 2, 1)

        b, c2, d, h, w = embedding.shape
        embedding = embedding.reshape(b, c2, d * h * w).permute(0, 2, 1)

        return embedding, compressed_embedding
