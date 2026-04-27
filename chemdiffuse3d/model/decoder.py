"""
Post-diffusion decoder architecture for ChemDiffuse3D.

Converts VAE latent outputs from the diffusion process into
full-resolution volumetric images, conditioned on the LR input.

Includes:
- AdaptedDecoder: Adapted decoder using residual channel attention and PixelShuffle upsampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ChannelAttention3D(nn.Module):
    def __init__(self, num_channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(num_channels, num_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_channels // reduction, num_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class RCAB3D(nn.Module):
    """Residual Channel Attention Block (3D)."""

    def __init__(self, num_channels, channel_reduction=8, residual_scaling=1.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(num_channels, num_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_channels, num_channels, 3, padding=1),
        )
        self.ca = ChannelAttention3D(num_channels, channel_reduction)
        self.residual_scaling = residual_scaling

    def forward(self, x):
        res = self.ca(self.body(x))
        if self.residual_scaling != 1.0:
            res = res * self.residual_scaling
        return x + res


class ResidualGroup3D(nn.Module):
    def __init__(self, num_channels, num_residual_blocks, channel_reduction=8, residual_scaling=1.0):
        super().__init__()
        self.rcabs = nn.Sequential(*[
            RCAB3D(num_channels, channel_reduction, residual_scaling)
            for _ in range(num_residual_blocks)
        ])
        self.conv = nn.Conv3d(num_channels, num_channels, 3, padding=1)

    def forward(self, x):
        return x + self.conv(self.rcabs(x))


class SpatialPixelShuffle3D(nn.Module):
    """PixelShuffle for spatial dimensions only (D is preserved)."""

    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.ps = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        B, C, D, H, W = x.size()
        x = x.transpose(1, 2).contiguous().view(B * D, C, H, W)
        x = self.ps(x)
        C_out = C // (self.upscale_factor ** 2)
        x = x.view(B, D, C_out, H * self.upscale_factor, W * self.upscale_factor)
        x = x.transpose(1, 2).contiguous()
        return x


class UpSampleBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super().__init__()
        assert upscale_factor in [2, 4, 8], "Only upscale factors 2, 4, 8 supported"
        ps_channels = out_channels * (upscale_factor ** 2)
        self.conv = nn.Conv3d(in_channels, ps_channels, 3, padding=1)
        self.ps = SpatialPixelShuffle3D(upscale_factor)
        self.rcab = RCAB3D(out_channels)

    def forward(self, x):
        return self.rcab(self.ps(self.conv(x)))


def get_lr_interpolated(lr_volume, vae_latent):
    """Interpolate LR volume depth to match VAE latent depth."""
    B, C, D_lr, H, W = lr_volume.shape
    D_hr = vae_latent.shape[2]
    if D_lr != D_hr:
        lr_volume = F.interpolate(lr_volume, size=(D_hr, H, W), mode='trilinear', align_corners=False)
    return lr_volume


class AdaptedDecoder(nn.Module):
    """
    Adapted decoder that upsamples VAE latents and fuses with LR volume
    via concatenation, followed by residual channel attention processing.
    """

    def __init__(self,
                 lr_channels=1,
                 latent_channels=4,
                 base_channels=32,
                 num_residual_blocks=3,
                 num_residual_groups=5):
        super().__init__()

        self.latent_up = nn.Sequential(
            UpSampleBlock3D(latent_channels, base_channels),
            UpSampleBlock3D(base_channels, base_channels),
            UpSampleBlock3D(base_channels, base_channels)
        )

        cat_channels = base_channels + lr_channels
        self.conv_first = nn.Conv3d(cat_channels, base_channels, 3, padding=1)

        self.groups = nn.ModuleList([
            ResidualGroup3D(base_channels, num_residual_blocks)
            for _ in range(num_residual_groups)
        ])

        self.conv_after_body = nn.Conv3d(base_channels, base_channels, 3, padding=1)
        self.conv_last = nn.Conv3d(base_channels, 1, 3, padding=1)

    def forward(self, lr_volume, vae_latent):
        lr_volume = get_lr_interpolated(lr_volume, vae_latent)

        latent_hr = self.latent_up(vae_latent)
        x = torch.cat([latent_hr, lr_volume], dim=1)

        x = self.conv_first(x)
        long_skip = x
        for group in self.groups:
            x = group(x)
        x = self.conv_after_body(x)
        x = x + long_skip
        x = self.conv_last(x)

        return x


