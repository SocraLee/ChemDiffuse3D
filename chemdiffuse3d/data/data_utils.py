"""
Data utilities for microscopy degradation simulation.

Provides physically-motivated PSF simulation and low-resolution
stack generation for training data augmentation.
"""

import numpy as np
from scipy.signal import fftconvolve
import torch


def resolution_sampler(nominal_na=1.05, wavelength=800e-9, n=1.406, underfillFactor=1.0):
    """
    Compute Gaussian-model PSF FWHM in lateral and axial dimensions.

    Parameters
    ----------
    nominal_na : float
        Nominal numerical aperture of the objective.
    wavelength : float
        Vacuum wavelength of the light (in meters).
    n : float
        Refractive index of the immersion medium.
    underfillFactor : float
        Ratio by which the back pupil is under-filled (NA_eff = nominal_na / k).

    Returns
    -------
    fwhm_xy_um : float
        Lateral Gaussian FWHM in microns.
    fwhm_z_um : float
        Axial Gaussian FWHM in microns.
    ratio : tuple of (1, float)
        Mapping ratio of axial FWHM vs. the no-underfill case.
    """
    na_eff = nominal_na / underfillFactor
    fwhm_xy = 0.51 * wavelength / na_eff
    fwhm_z = 0.88 * n * wavelength / (na_eff ** 2)
    fwhm_z_nom = 0.88 * n * wavelength / (nominal_na ** 2)
    factor = fwhm_z / fwhm_z_nom
    return fwhm_xy * 1e6, fwhm_z * 1e6, (1, factor)


def simulate_low_resolution_stack(high_res_stack, psf, z_ratio=1, normalize_psf=False):
    """
    Convolve a 3D image stack with a measured PSF and downsample in Z.

    Parameters
    ----------
    high_res_stack : ndarray, shape (Z, Y, X)
    psf : ndarray, shape (pz, py, px)
    z_ratio : int
        Downsampling factor along Z axis.
    normalize_psf : bool
        Whether to normalize PSF to unit energy.

    Returns
    -------
    low_res_stack : ndarray
    """
    if normalize_psf:
        psf = psf / np.sum(psf)
    return fftconvolve(high_res_stack, psf, mode='same')[z_ratio // 2::z_ratio]


def generate_psf_from_fwhm(psf_shape, voxel_size, fwhm_xy_um, fwhm_z_um):
    """
    Generate a 3D Gaussian PSF whose FWHM matches specified values.

    Parameters
    ----------
    psf_shape : tuple of ints (Z, Y, X)
    voxel_size : tuple of floats (dz, dy, dx) in microns
    fwhm_xy_um : float, lateral FWHM in microns
    fwhm_z_um : float, axial FWHM in microns

    Returns
    -------
    psf : ndarray (Z, Y, X), normalized to sum = 1
    """
    dz, dy, dx = voxel_size
    z_sz, y_sz, x_sz = psf_shape
    fwhm2sigma = 1.0 / (2 * np.sqrt(2 * np.log(2)))
    sigma_xy_um = fwhm_xy_um * fwhm2sigma
    sigma_z_um = fwhm_z_um * fwhm2sigma
    sigma_z = sigma_z_um / dz
    sigma_y = sigma_xy_um / dy
    sigma_x = sigma_xy_um / dx

    zz, yy, xx = np.meshgrid(
        np.arange(z_sz) - z_sz // 2,
        np.arange(y_sz) - y_sz // 2,
        np.arange(x_sz) - x_sz // 2,
        indexing='ij'
    )
    psf = np.exp(
        -(xx ** 2) / (2 * sigma_x ** 2)
        - (yy ** 2) / (2 * sigma_y ** 2)
        - (zz ** 2) / (2 * sigma_z ** 2)
    )
    psf /= psf.sum()
    return psf


class RandomMicroscopeDegradation:
    """
    Simulate microscopy degradation by randomly sampling a PSF
    based on a target Z-axis downsampling ratio.

    The underfill factor is computed as sqrt(ratio), which maps
    the desired resolution degradation to an effective numerical
    aperture reduction: FWHM_z ~ (underfill)^2.

    Args:
        voxel_size (tuple): HR image voxel size (dz, dy, dx) in microns.
        downsample_ratios (list): Candidate Z-axis downsampling factors.
        nominal_na (float): Nominal numerical aperture of the objective.
        wavelength (float): Light wavelength in meters.
        n_refractive (float): Refractive index.
        cache_psf (bool): Whether to pre-compute and cache PSFs.
    """

    def __init__(self,
                 voxel_size=(1.0, 0.4077, 0.4077),
                 downsample_ratios=[4],
                 nominal_na=1.05,
                 wavelength=800e-9,
                 n_refractive=1.406,
                 cache_psf=True):

        self.voxel_size = voxel_size
        self.ratios = downsample_ratios
        self.nominal_na = nominal_na
        self.wavelength = wavelength
        self.n = n_refractive
        self.cache_psf = cache_psf
        self.psf_bank = {}

        if self.cache_psf:
            for r in self.ratios:
                self.psf_bank[r] = self._generate_params_for_ratio(r)

    def _generate_params_for_ratio(self, target_z_ratio):
        """Generate PSF for a given Z-axis downsampling ratio."""
        u_factor = np.sqrt(target_z_ratio)

        fwhm_xy, fwhm_z, _ = resolution_sampler(
            nominal_na=self.nominal_na,
            wavelength=self.wavelength,
            n=self.n,
            underfillFactor=u_factor
        )

        dz, dy, dx = self.voxel_size
        fwhm2sigma = 1.0 / (2 * np.sqrt(2 * np.log(2)))
        sigma_z_px = (fwhm_z * fwhm2sigma) / dz
        sigma_xy_px = (fwhm_xy * fwhm2sigma) / dy

        def odd(x):
            return int(np.ceil(x)) // 2 * 2 + 1

        psf_size_factor = 6
        pz = odd(psf_size_factor * sigma_z_px)
        pxy = odd(psf_size_factor * sigma_xy_px)
        psf_shape = (pz, pxy, pxy)
        psf = generate_psf_from_fwhm(psf_shape, self.voxel_size, fwhm_xy, fwhm_z)
        return psf

    def __call__(self, hr_stack):
        """
        Apply random microscopy degradation.

        Args:
            hr_stack (ndarray or Tensor): Shape (Z, Y, X).

        Returns:
            lr_stack, chosen_ratio
        """
        idx = torch.randint(0, len(self.ratios), (1,)).item()
        z_ratio = self.ratios[idx]

        if self.cache_psf:
            psf = self.psf_bank[z_ratio]
        else:
            psf = self._generate_params_for_ratio(z_ratio)

        is_tensor = False
        if isinstance(hr_stack, torch.Tensor):
            is_tensor = True
            device = hr_stack.device
            hr_stack = hr_stack.cpu().numpy()

        assert hr_stack.ndim == 3
        lr_stack = simulate_low_resolution_stack(hr_stack, psf, z_ratio=z_ratio)

        if is_tensor:
            lr_stack = torch.from_numpy(lr_stack).to(device).float()

        return lr_stack, z_ratio

    def __repr__(self):
        return f"{self.__class__.__name__}(ratios={self.ratios}, voxel_size={self.voxel_size})"
