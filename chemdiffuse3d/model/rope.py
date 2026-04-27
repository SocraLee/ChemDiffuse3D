"""
RoPE (Rotary Position Embedding) for 1D and 2D grids.

Provides frequency-based positional encoding that can be applied
to query and key tensors in attention layers. Not currently used
in the default architecture but available for experimentation.
"""

import torch
import torch.nn as nn
import math
from torch import Tensor


class RopePositionEmbedding2D(nn.Module):
    """2D Rotary Position Embedding for spatial (H, W) attention."""

    def __init__(self, *, D_head: int, base: float = 100.0, **kwargs):
        super().__init__()
        assert D_head % 4 == 0
        self.D_head = D_head
        self.base = base
        self.dtype = kwargs.get("dtype", torch.float32)

        self.register_buffer(
            "periods",
            torch.empty(D_head // 4, dtype=self.dtype),
            persistent=True,
        )
        self._init_weights()

    def _init_weights(self):
        device = self.periods.device
        self.periods.data = self.base ** (
                2 * torch.arange(self.D_head // 4, device=device, dtype=self.dtype) / (self.D_head // 2)
        )

    def forward(self, *, H: int, W: int) -> tuple[Tensor, Tensor]:
        """
        Returns:
            tuple[Tensor, Tensor]: (sin, cos) of shape [HW, D_head]
        """
        device = self.periods.device
        dd = {"device": device, "dtype": self.dtype}

        coords_h = torch.arange(H, **dd)
        coords_w = torch.arange(W, **dd)

        coords_grid = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        coords = coords_grid.flatten(0, 1)  # [HW, 2]

        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2)  # [HW, D_head // 2]
        angles = angles.tile(2)  # [HW, D_head]

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        return (sin, cos)


class RopePositionEmbedding1D(nn.Module):
    """1D Rotary Position Embedding for depth (D) attention."""

    def __init__(self, *, D_head: int, base: float = 100.0, **kwargs):
        super().__init__()
        assert D_head % 2 == 0
        self.D_head = D_head
        self.base = base

        self.dtype = kwargs.get("dtype", torch.float32)
        self.register_buffer(
            "periods",
            torch.empty(D_head // 2, dtype=self.dtype),
            persistent=True,
        )
        self._init_weights()

    def _init_weights(self):
        self.periods.data = self.base ** (
            torch.arange(0, self.D_head, 2, dtype=self.dtype) / self.D_head
        )

    def forward(self, *, D: int) -> tuple[Tensor, Tensor]:
        device = self.periods.device
        dd = {"device": device, "dtype": self.dtype}

        coords_d = torch.arange(D, **dd)
        angles = 2 * math.pi * coords_d[:, None] / self.periods[None, :]
        angles = angles.repeat(1, 2)

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        return (sin, cos)
