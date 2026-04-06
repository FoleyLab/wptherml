"""Gradient helpers for wptherml2."""

from __future__ import annotations

import numpy as np

from ..types import GradientResult, TmmRequest
from .solve import solve_tmm


def compute_thickness_gradients(request: TmmRequest) -> GradientResult:
    """Compute spectral thickness gradients using centered finite differences.

    This is a correctness-first baseline for the new package scaffold. The API
    is intended to remain stable when analytical thickness derivatives replace
    the internal implementation.
    """

    layers = np.asarray(request.gradient_layers, dtype=np.int64)
    n_wavelengths = request.grid.wavelength_m.shape[0]
    d_reflectivity = np.zeros((layers.shape[0], n_wavelengths), dtype=np.float64)
    d_transmissivity = np.zeros_like(d_reflectivity)
    d_absorptivity = np.zeros_like(d_reflectivity)

    for row, layer in enumerate(layers):
        thickness_plus = request.stack.thickness_m.copy()
        thickness_minus = request.stack.thickness_m.copy()
        thickness_plus[layer] += request.gradient_step_m
        thickness_minus[layer] -= request.gradient_step_m

        result_plus = solve_tmm(request.with_thickness(thickness_plus))
        result_minus = solve_tmm(request.with_thickness(thickness_minus))

        scale = 2.0 * request.gradient_step_m
        d_reflectivity[row] = (result_plus.reflectivity - result_minus.reflectivity) / scale
        d_transmissivity[row] = (result_plus.transmissivity - result_minus.transmissivity) / scale
        d_absorptivity[row] = (result_plus.absorptivity - result_minus.absorptivity) / scale

    return GradientResult(
        wavelength_m=request.grid.wavelength_m,
        gradient_layers=layers,
        d_reflectivity=d_reflectivity,
        d_transmissivity=d_transmissivity,
        d_absorptivity=d_absorptivity,
    )
