"""Typed input and output containers for wptherml2."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.float64]
ArrayC = NDArray[np.complex128]


def _as_float_array(values: Sequence[float]) -> ArrayF:
    return np.asarray(values, dtype=np.float64)


def _as_complex_matrix(values: Sequence[Sequence[complex]]) -> ArrayC:
    return np.asarray(values, dtype=np.complex128)


@dataclass(frozen=True)
class LayerStack:
    """Geometry and optional labels for a multilayer stack."""

    thickness_m: ArrayF
    materials: Sequence[str] | None = None

    def __post_init__(self) -> None:
        thickness_m = _as_float_array(self.thickness_m)
        if thickness_m.ndim != 1:
            raise ValueError("LayerStack.thickness_m must be one-dimensional.")
        if thickness_m.shape[0] < 2:
            raise ValueError("LayerStack must include at least incident and exit media.")
        object.__setattr__(self, "thickness_m", thickness_m)

        if self.materials is not None and len(self.materials) != thickness_m.shape[0]:
            raise ValueError("LayerStack.materials must match thickness_m length.")


@dataclass(frozen=True)
class SpectralGrid:
    """Spectral sampling and single-angle illumination settings."""

    wavelength_m: ArrayF
    incident_angle_rad: float = 0.0
    polarization: str = "p"

    def __post_init__(self) -> None:
        wavelength_m = _as_float_array(self.wavelength_m)
        if wavelength_m.ndim != 1:
            raise ValueError("SpectralGrid.wavelength_m must be one-dimensional.")
        if np.any(wavelength_m <= 0.0):
            raise ValueError("SpectralGrid.wavelength_m must be strictly positive.")
        if self.polarization not in {"s", "p"}:
            raise ValueError("SpectralGrid.polarization must be 's' or 'p'.")
        object.__setattr__(self, "wavelength_m", wavelength_m)


@dataclass(frozen=True)
class TmmRequest:
    """Input bundle for one-angle TMM calculations."""

    stack: LayerStack
    grid: SpectralGrid
    refractive_index_nk: ArrayC
    backend: str = "auto"
    gradient_layers: ArrayF | None = None
    gradient_step_m: float = 1.0e-12

    def __post_init__(self) -> None:
        refractive_index_nk = _as_complex_matrix(self.refractive_index_nk)
        n_wavelengths = self.grid.wavelength_m.shape[0]
        n_layers = self.stack.thickness_m.shape[0]

        if refractive_index_nk.shape != (n_wavelengths, n_layers):
            raise ValueError(
                "TmmRequest.refractive_index_nk must have shape "
                f"({n_wavelengths}, {n_layers})."
            )
        if self.backend not in {"auto", "scalar", "vectorized"}:
            raise ValueError("TmmRequest.backend must be 'auto', 'scalar', or 'vectorized'.")
        if self.gradient_step_m <= 0.0:
            raise ValueError("TmmRequest.gradient_step_m must be positive.")

        object.__setattr__(self, "refractive_index_nk", refractive_index_nk)

        if self.gradient_layers is None:
            layers = np.arange(1, max(1, n_layers - 1), dtype=np.int64)
        else:
            layers = np.asarray(self.gradient_layers, dtype=np.int64)

        if layers.ndim != 1:
            raise ValueError("TmmRequest.gradient_layers must be one-dimensional.")
        if layers.size > 0:
            if np.any(layers <= 0) or np.any(layers >= n_layers - 1):
                raise ValueError("Gradient layers must refer to interior layers only.")

        object.__setattr__(self, "gradient_layers", layers)

    def with_thickness(self, thickness_m: ArrayF) -> "TmmRequest":
        """Return a new request with updated stack thickness."""

        return replace(self, stack=replace(self.stack, thickness_m=_as_float_array(thickness_m)))


@dataclass(frozen=True)
class KernelState:
    """Optional diagnostic outputs from the TMM kernel."""

    k0: ArrayF
    kx: ArrayC
    kz: ArrayC
    cos_theta: ArrayC
    transfer_matrix: ArrayC
    backend: str


@dataclass(frozen=True)
class SpectrumResult:
    """Wavelength-resolved TMM observables."""

    wavelength_m: ArrayF
    reflectivity: ArrayF
    transmissivity: ArrayF
    absorptivity: ArrayF
    state: KernelState | None = None


@dataclass(frozen=True)
class GradientResult:
    """Thickness gradients for spectral observables."""

    wavelength_m: ArrayF
    gradient_layers: ArrayF
    d_reflectivity: ArrayF
    d_transmissivity: ArrayF
    d_absorptivity: ArrayF
