"""Core typed containers used across wptherml modules.

This module introduces lightweight dataclasses to represent simulation requests
and standardized result objects.  The intent is to decouple physics kernels
from legacy driver classes and support progressive API migration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.float64]
ArrayC = NDArray[np.complex128]


@dataclass(frozen=True)
class LayerStack:
    """Layer-stack geometry and material identity.

    Parameters
    ----------
    thickness_m : numpy.ndarray
        One-dimensional array of layer thicknesses in meters with shape
        ``(number_of_layers,)``.
    materials : Sequence[str]
        Material labels for each layer.  The length should match
        ``thickness_m``.
    """

    thickness_m: ArrayF
    materials: Sequence[str]


@dataclass(frozen=True)
class SpectralGrid:
    """Spectral and angular sampling definition.

    Parameters
    ----------
    wavelength_m : numpy.ndarray
        One-dimensional wavelength grid in meters.
    incident_angle_rad : float, optional
        Incident angle in radians used for single-angle calculations.
    polarization : str, optional
        Input polarization (typically ``"s"`` or ``"p"``).
    theta_vals_rad : numpy.ndarray, optional
        Explicit angle grid for angle-resolved calculations.
    theta_weights : numpy.ndarray, optional
        Quadrature weights corresponding to ``theta_vals_rad``.
    """

    wavelength_m: ArrayF
    incident_angle_rad: float = 0.0
    polarization: str = "p"
    theta_vals_rad: Optional[ArrayF] = None
    theta_weights: Optional[ArrayF] = None


@dataclass(frozen=True)
class TmmSimulationRequest:
    """Input definition for a TMM observable calculation.

    Parameters
    ----------
    stack : LayerStack
        Layer-stack specification.
    grid : SpectralGrid
        Spectral and angular sampling parameters.
    refractive_index_nk : numpy.ndarray, optional
        Precomputed complex refractive-index array with shape
        ``(number_of_wavelengths, number_of_layers)``.  If omitted, callers are
        expected to provide material models externally.
    backend : str, optional
        Backend selection hint.  Supported values are ``"auto"``, ``"scalar"``,
        and ``"vectorized"``.
    compute_gradient : bool, optional
        Flag indicating whether gradients should be computed.
    gradient_layers : numpy.ndarray, optional
        Layers to include in gradient calculations.
    """

    stack: LayerStack
    grid: SpectralGrid
    refractive_index_nk: Optional[ArrayC] = None
    backend: str = "auto"
    compute_gradient: bool = False
    gradient_layers: Optional[ArrayF] = None


@dataclass
class SpectrumResult:
    """Wavelength-resolved optical observables.

    Parameters
    ----------
    reflectivity : numpy.ndarray
        Reflectivity array versus wavelength.
    transmissivity : numpy.ndarray
        Transmissivity array versus wavelength.
    emissivity : numpy.ndarray
        Emissivity array versus wavelength.
    wavelength_m : numpy.ndarray
        Wavelength grid used for the calculation.
    """

    reflectivity: ArrayF
    transmissivity: ArrayF
    emissivity: ArrayF
    wavelength_m: ArrayF


@dataclass
class AngleResolvedResult:
    """Angle- and wavelength-resolved optical observables."""

    reflectivity_s: Optional[ArrayF] = None
    reflectivity_p: Optional[ArrayF] = None
    transmissivity_s: Optional[ArrayF] = None
    transmissivity_p: Optional[ArrayF] = None
    emissivity_s: Optional[ArrayF] = None
    emissivity_p: Optional[ArrayF] = None
    theta_vals_rad: Optional[ArrayF] = None
    wavelength_m: Optional[ArrayF] = None


@dataclass
class TmmKernelState:
    """Optional diagnostic arrays from core kernel execution."""

    kz: Optional[ArrayC] = None
    k0: Optional[ArrayF] = None
    kx: Optional[ArrayC] = None
    transfer_matrix: Optional[ArrayC] = None


@dataclass
class TmmGradientResult:
    """Gradient arrays for spectral observables."""

    d_reflectivity: Optional[ArrayF] = None
    d_transmissivity: Optional[ArrayF] = None
    d_emissivity: Optional[ArrayF] = None
    gradient_layers: Optional[ArrayF] = None


@dataclass
class TmmObservablesResult:
    """Unified container for TMM outputs.

    Parameters
    ----------
    spectrum : SpectrumResult, optional
        Wavelength-resolved observables.
    angle_resolved : AngleResolvedResult, optional
        Angle-resolved observables.
    gradient : TmmGradientResult, optional
        Gradient outputs.
    state : TmmKernelState, optional
        Kernel diagnostic arrays.
    metadata : dict, optional
        Additional metadata for provenance and backend reporting.
    """

    spectrum: Optional[SpectrumResult] = None
    angle_resolved: Optional[AngleResolvedResult] = None
    gradient: Optional[TmmGradientResult] = None
    state: Optional[TmmKernelState] = None
    metadata: Dict[str, str] = field(default_factory=dict)
