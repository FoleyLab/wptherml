"""Wavevector helpers for transfer-matrix calculations."""

from __future__ import annotations

import numpy as np

from ..types import ArrayC, ArrayF


def compute_k0(wavelength_m: ArrayF) -> ArrayF:
    """Compute free-space wavevector magnitude."""

    return 2.0 * np.pi / wavelength_m


def compute_kx(refractive_index_incident: ArrayC, k0: ArrayF, incident_angle_rad: float) -> ArrayC:
    """Compute the in-plane wavevector component."""

    return refractive_index_incident * np.sin(incident_angle_rad) * k0


def compute_kz(refractive_index_nk: ArrayC, k0: ArrayF, kx: ArrayC) -> ArrayC:
    """Compute the normal wavevector in each layer."""

    return np.sqrt((refractive_index_nk * k0[:, None]) ** 2 - kx[:, None] ** 2)


def compute_cos_theta(refractive_index_nk: ArrayC, k0: ArrayF, kz: ArrayC, incident_angle_rad: float) -> ArrayC:
    """Compute cosine of the propagation angle in each layer."""

    cos_theta = np.zeros_like(refractive_index_nk, dtype=np.complex128)
    cos_theta[:, 0] = np.cos(incident_angle_rad)
    cos_theta[:, 1:] = kz[:, 1:] / (refractive_index_nk[:, 1:] * k0[:, None])
    return cos_theta
