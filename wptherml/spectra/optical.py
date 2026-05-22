"""Optical spectrum result containers."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class OpticalSpectrum:
    wavelengths: NDArray
    reflectivity: NDArray
    transmissivity: NDArray
    emissivity: NDArray
    angles: NDArray | None = None
    polarizations: list[str] | None = None

    def __post_init__(self) -> None:
        wavelengths = np.asarray(self.wavelengths, dtype=float)
        reflectivity = np.asarray(self.reflectivity, dtype=float)
        transmissivity = np.asarray(self.transmissivity, dtype=float)
        emissivity = np.asarray(self.emissivity, dtype=float)
        angles = (
            None
            if self.angles is None
            else np.atleast_1d(np.asarray(self.angles, dtype=float))
        )
        polarizations = (
            None if self.polarizations is None else list(self.polarizations)
        )

        if wavelengths.ndim != 1:
            raise ValueError("wavelengths must be a one-dimensional array")
        if reflectivity.shape != transmissivity.shape:
            raise ValueError("reflectivity and transmissivity shapes must match")
        if reflectivity.shape != emissivity.shape:
            raise ValueError("reflectivity and emissivity shapes must match")
        if reflectivity.shape[0] != len(wavelengths):
            raise ValueError("spectrum arrays must use wavelengths as axis 0")

        object.__setattr__(self, "wavelengths", wavelengths)
        object.__setattr__(self, "reflectivity", reflectivity)
        object.__setattr__(self, "transmissivity", transmissivity)
        object.__setattr__(self, "emissivity", emissivity)
        object.__setattr__(self, "angles", angles)
        object.__setattr__(self, "polarizations", polarizations)


@dataclass(frozen=True)
class OpticalSpectrumGradient:
    dR_dd: NDArray
    dT_dd: NDArray
    dE_dd: NDArray

    def __post_init__(self) -> None:
        dR_dd = np.asarray(self.dR_dd, dtype=float)
        dT_dd = np.asarray(self.dT_dd, dtype=float)
        dE_dd = np.asarray(self.dE_dd, dtype=float)

        if dR_dd.shape != dT_dd.shape:
            raise ValueError("dR_dd and dT_dd shapes must match")
        if dR_dd.shape != dE_dd.shape:
            raise ValueError("dR_dd and dE_dd shapes must match")

        object.__setattr__(self, "dR_dd", dR_dd)
        object.__setattr__(self, "dT_dd", dT_dd)
        object.__setattr__(self, "dE_dd", dE_dd)
