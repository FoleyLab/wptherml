"""Observable assembly from transfer matrices."""

from __future__ import annotations

import numpy as np

from ..types import ArrayC


def compute_observables(transfer_matrix: ArrayC, refractive_index_nk: ArrayC, cos_theta: ArrayC) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute spectral reflectivity, transmissivity, and absorptivity."""

    reflection = transfer_matrix[:, 1, 0] / transfer_matrix[:, 0, 0]
    transmission = 1.0 / transfer_matrix[:, 0, 0]
    factor = refractive_index_nk[:, -1] * cos_theta[:, -1] / (refractive_index_nk[:, 0] * cos_theta[:, 0])

    reflectivity = np.real(reflection * np.conjugate(reflection))
    transmissivity = np.real(transmission * np.conjugate(transmission) * factor)
    absorptivity = 1.0 - reflectivity - transmissivity
    return reflectivity, transmissivity, absorptivity
