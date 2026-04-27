"""
SILoss: Scale-Invariant Loss for flow-matching diffusion training.

Combines denoising loss with optional REPA (REPresentation Alignment)
projection loss for improved generation quality.
"""

import torch
import numpy as np
import torch.nn.functional as F


def mean_flat(x):
    """Take the mean over all non-batch dimensions."""
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def sum_flat(x):
    """Take the sum over all non-batch dimensions."""
    return torch.sum(x, dim=list(range(1, len(x.size()))))


class SILoss:
    """
    Scale-Invariant Loss combining denoising and representation alignment.

    Args:
        prediction: Prediction target type ('v' for velocity)
        path_type: Interpolation path type ('linear' or 'cosine')
        weighting: Timestep sampling strategy ('uniform' or 'lognormal')
        accelerator: HuggingFace Accelerator instance
        z_alignment_weights: Per-type weights for REPA loss
        z_types_in_model_output: Which z-types the model produces
    """

    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[],
            accelerator=None,
            z_alignment_weights={'dinov2': 0.15, 'sam': 0.8, 'clip': 0.05},
            z_types_in_model_output=['dinov2', 'sam', 'clip'],
    ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.z_alignment_weights = z_alignment_weights
        self.z_types_in_model_output = z_types_in_model_output

    def interpolant(self, t):
        """Compute interpolation coefficients for the given path type."""
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t = 1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t = np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, model, images, model_kwargs=None, zs=None):
        # Sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1, 1))
        elif self.weighting == "lognormal":
            rnd_normal = torch.randn((images.shape[0], 1, 1, 1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)

        time_input = time_input.to(device=images.device, dtype=images.dtype)

        # Construct noisy input
        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)

        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError()

        # Forward pass
        model_output, zs_tilde = model(model_input, time_input.flatten(), **model_kwargs)
        denoising_loss = mean_flat((model_output - model_target) ** 2)

        # REPA projection loss
        proj_loss_total = torch.tensor(0.0, device=images.device, dtype=images.dtype)

        if zs and zs_tilde is not None:
            active_weights_sum = 0.0
            for key in zs_tilde:
                current_z_type = key
                weight = self.z_alignment_weights.get(current_z_type)
                z_pred = zs_tilde[key]
                z_target = zs[key]

                if weight == 0.0:
                    continue

                if z_target.ndim == 4:  # DINOv2 patch embeddings (B, D, T, H)
                    B, D, T, H = z_target.shape
                    z_target = z_target.reshape(B, D * T, H)
                    z_pred_norm = F.normalize(z_pred, dim=-1)
                    z_target_norm = F.normalize(z_target, dim=-1)
                    cos_sim = (z_pred_norm * z_target_norm).sum(dim=-1)
                    current_type_loss = -cos_sim.mean()
                else:
                    print(f"Warning: Unsupported z_target ndim {z_target.ndim} for type {current_z_type}")
                    current_type_loss = torch.tensor(0.0, device=images.device, dtype=images.dtype)

                proj_loss_total += weight * current_type_loss
                active_weights_sum += weight

            if active_weights_sum > 0:
                proj_loss = proj_loss_total / active_weights_sum
            else:
                raise ValueError("No active REPA alignment weights found.")
        else:
            proj_loss = torch.tensor(0.0, device=images.device, dtype=images.dtype)

        return denoising_loss, proj_loss
