"""Observable assembly helpers for transfer-matrix outputs."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from wptherml.core.types import SpectrumResult

ArrayC = NDArray[np.complex128]


def spectrum_from_transfer_matrix(
    tm: ArrayC,
    refractive_index: ArrayC,
    cos_theta: ArrayC,
    wavelength_m: NDArray[np.float64],
) -> SpectrumResult:
    """Build reflectivity/transmissivity/emissivity from transfer matrices.

    Parameters
    ----------
    tm : numpy.ndarray
        Transfer matrix array with shape ``(number_of_wavelengths, 2, 2)``.
    refractive_index : numpy.ndarray
        Complex refractive index array with shape
        ``(number_of_wavelengths, number_of_layers)``.
    cos_theta : numpy.ndarray
        Cosine-angle array with shape ``(number_of_wavelengths, number_of_layers)``.
    wavelength_m : numpy.ndarray
        Wavelength grid in meters.

    Returns
    -------
    SpectrumResult
        Structured spectral observables.
    """

    r = tm[:, 1, 0] / tm[:, 0, 0]
    t = 1.0 / tm[:, 0, 0]
    factor = (
        refractive_index[:, -1] * cos_theta[:, -1] /
        (refractive_index[:, 0] * cos_theta[:, 0])
    )

    reflectivity = np.real(r * np.conj(r))
    transmissivity = np.real(t * np.conj(t) * factor)
    emissivity = 1.0 - reflectivity - transmissivity
    return SpectrumResult(reflectivity, transmissivity, emissivity, wavelength_m)
