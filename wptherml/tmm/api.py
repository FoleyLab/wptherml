"""High-level API wrappers around TMM kernel functions."""

from __future__ import annotations

from wptherml.core.types import (
    TmmKernelState,
    TmmObservablesResult,
    TmmSimulationRequest,
)

from .kernels import compute_k0, compute_kx, compute_kz, compute_transfer_matrix
from .observables import spectrum_from_transfer_matrix


def compute_observables(request: TmmSimulationRequest) -> TmmObservablesResult:
    """Compute TMM observables from a structured request object.

    Parameters
    ----------
    request : TmmSimulationRequest
        Structured inputs for transfer-matrix evaluation.

    Returns
    -------
    TmmObservablesResult
        Structured output containing spectrum data and optional state metadata.

    Raises
    ------
    ValueError
        If no refractive index data is available in the request.
    """

    refractive_index = request.refractive_index_nk
    if refractive_index is None:
        raise ValueError(
            "TmmSimulationRequest.refractive_index_nk is required for this API "
            "path. Populate refractive_index_nk or use a higher-level driver "
            "that constructs material optical data."
        )

    wavelength = request.grid.wavelength_m
    angle = request.grid.incident_angle_rad
    polarization = request.grid.polarization

    k0 = compute_k0(wavelength)
    kx = compute_kx(refractive_index[:, 0], k0, angle)
    kz = compute_kz(refractive_index, k0, kx)
    tm, cos_theta = compute_transfer_matrix(
        refractive_index,
        k0,
        kz,
        request.stack.thickness_m,
        angle,
        polarization,
        request.backend,
    )

    spectrum = spectrum_from_transfer_matrix(tm, refractive_index, cos_theta, wavelength)
    state = TmmKernelState(kz=kz, k0=k0, kx=kx, transfer_matrix=tm)
    metadata = {"backend": request.backend, "polarization": polarization}
    return TmmObservablesResult(spectrum=spectrum, state=state, metadata=metadata)
