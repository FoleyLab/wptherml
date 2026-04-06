"""Forward TMM solve routines."""

from __future__ import annotations

import numpy as np

from ..types import ArrayC, KernelState, SpectrumResult, TmmRequest
from .matrices import compute_d_matrix, compute_propagation_matrices, compute_propagation_matrix
from .observables import compute_observables
from .wavevectors import compute_cos_theta, compute_k0, compute_kx, compute_kz


def _resolve_backend(request: TmmRequest) -> str:
    if request.backend == "auto":
        return "vectorized"
    return request.backend


def _solve_scalar(request: TmmRequest, k0: np.ndarray, kz: ArrayC) -> tuple[ArrayC, ArrayC]:
    n_wavelengths, n_layers = request.refractive_index_nk.shape
    transfer_matrix = np.zeros((n_wavelengths, 2, 2), dtype=np.complex128)
    cos_theta = np.zeros((n_wavelengths, n_layers), dtype=np.complex128)
    thickness_m = request.stack.thickness_m

    for idx in range(n_wavelengths):
        refractive_index = request.refractive_index_nk[idx]
        kz_row = kz[idx]
        cos_theta[idx, 0] = np.cos(request.grid.incident_angle_rad)
        cos_theta[idx, 1:] = kz_row[1:] / (refractive_index[1:] * k0[idx])
        phase = kz_row * thickness_m

        _, running = compute_d_matrix(refractive_index[0], cos_theta[idx, 0], request.grid.polarization)
        for layer in range(1, n_layers - 1):
            d_matrix, d_inverse = compute_d_matrix(refractive_index[layer], cos_theta[idx, layer], request.grid.polarization)
            running = running @ d_matrix @ compute_propagation_matrix(phase[layer]) @ d_inverse

        last_matrix, _ = compute_d_matrix(refractive_index[-1], cos_theta[idx, -1], request.grid.polarization)
        transfer_matrix[idx] = running @ last_matrix

    return transfer_matrix, cos_theta


def _solve_vectorized(request: TmmRequest, k0: np.ndarray, kz: ArrayC) -> tuple[ArrayC, ArrayC]:
    refractive_index_nk = request.refractive_index_nk
    n_wavelengths, n_layers = refractive_index_nk.shape
    thickness_m = request.stack.thickness_m
    cos_theta = compute_cos_theta(refractive_index_nk, k0, kz, request.grid.incident_angle_rad)

    transfer_matrix = np.zeros((n_wavelengths, 2, 2), dtype=np.complex128)
    for idx in range(n_wavelengths):
        _, incident_inverse = compute_d_matrix(
            refractive_index_nk[idx, 0],
            cos_theta[idx, 0],
            request.grid.polarization,
        )
        transfer_matrix[idx] = incident_inverse

    for layer in range(1, n_layers - 1):
        d_matrices = np.zeros((n_wavelengths, 2, 2), dtype=np.complex128)
        d_inverses = np.zeros((n_wavelengths, 2, 2), dtype=np.complex128)
        for idx in range(n_wavelengths):
            d_matrix, d_inverse = compute_d_matrix(
                refractive_index_nk[idx, layer],
                cos_theta[idx, layer],
                request.grid.polarization,
            )
            d_matrices[idx] = d_matrix
            d_inverses[idx] = d_inverse

        phase = kz[:, layer] * thickness_m[layer]
        propagation = compute_propagation_matrices(phase)
        transfer_matrix = transfer_matrix @ d_matrices
        transfer_matrix = transfer_matrix @ propagation
        transfer_matrix = transfer_matrix @ d_inverses

    last_matrices = np.zeros((n_wavelengths, 2, 2), dtype=np.complex128)
    for idx in range(n_wavelengths):
        d_matrix, _ = compute_d_matrix(
            refractive_index_nk[idx, -1],
            cos_theta[idx, -1],
            request.grid.polarization,
        )
        last_matrices[idx] = d_matrix
    transfer_matrix = transfer_matrix @ last_matrices
    return transfer_matrix, cos_theta


def solve_tmm(request: TmmRequest) -> SpectrumResult:
    """Run the forward TMM solver and return spectral observables."""

    backend = _resolve_backend(request)
    k0 = compute_k0(request.grid.wavelength_m)
    kx = compute_kx(request.refractive_index_nk[:, 0], k0, request.grid.incident_angle_rad)
    kz = compute_kz(request.refractive_index_nk, k0, kx)

    if backend == "scalar":
        transfer_matrix, cos_theta = _solve_scalar(request, k0, kz)
    elif backend == "vectorized":
        transfer_matrix, cos_theta = _solve_vectorized(request, k0, kz)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    reflectivity, transmissivity, absorptivity = compute_observables(
        transfer_matrix,
        request.refractive_index_nk,
        cos_theta,
    )
    state = KernelState(
        k0=k0,
        kx=kx,
        kz=kz,
        cos_theta=cos_theta,
        transfer_matrix=transfer_matrix,
        backend=backend,
    )
    return SpectrumResult(
        wavelength_m=request.grid.wavelength_m,
        reflectivity=reflectivity,
        transmissivity=transmissivity,
        absorptivity=absorptivity,
        state=state,
    )
