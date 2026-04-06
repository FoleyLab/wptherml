"""Matrix assembly utilities for scalar and vectorized TMM paths."""

from __future__ import annotations

import numpy as np

from ..types import ArrayC


def compute_d_matrix(refractive_index: complex, cosine_theta: complex, polarization: str) -> tuple[ArrayC, ArrayC]:
    """Construct the interface matrix and its inverse for one layer."""

    d_matrix = np.zeros((2, 2), dtype=np.complex128)
    if polarization == "s":
        admittance = refractive_index * cosine_theta
        d_matrix[0, 0] = 1.0
        d_matrix[0, 1] = 1.0
        d_matrix[1, 0] = admittance
        d_matrix[1, 1] = -admittance
    else:
        d_matrix[0, 0] = cosine_theta
        d_matrix[0, 1] = cosine_theta
        d_matrix[1, 0] = refractive_index
        d_matrix[1, 1] = -refractive_index

    d_inverse = np.linalg.inv(d_matrix)
    return d_matrix, d_inverse


def compute_propagation_matrix(phi: complex) -> ArrayC:
    """Construct the propagation matrix for one layer."""

    propagation = np.eye(2, dtype=np.complex128)
    propagation[0, 0] = np.exp(-1j * phi)
    propagation[1, 1] = np.exp(1j * phi)
    return propagation


def compute_propagation_matrices(phi: ArrayC) -> ArrayC:
    """Construct propagation matrices across all wavelengths for one layer."""

    propagation = np.zeros((phi.shape[0], 2, 2), dtype=np.complex128)
    propagation[:, 0, 0] = np.exp(-1j * phi)
    propagation[:, 1, 1] = np.exp(1j * phi)
    return propagation
