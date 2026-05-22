"""Selective mirror objective functions."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .base import Objective
from ..spectra import OpticalSpectrum, OpticalSpectrumGradient


@dataclass(frozen=True)
class SelectiveMirrorObjective(Objective):
    transmissive_envelope: NDArray
    reflective_envelope: NDArray
    transmission_efficiency_weight: float = 1.0 / 3.0
    reflection_efficiency_weight: float = 1.0 / 3.0
    reflection_selectivity_weight: float = 1.0 / 3.0

    def __post_init__(self) -> None:
        transmissive_envelope = np.asarray(self.transmissive_envelope, dtype=float)
        reflective_envelope = np.asarray(self.reflective_envelope, dtype=float)
        if transmissive_envelope.shape != reflective_envelope.shape:
            raise ValueError("transmissive and reflective envelopes must match")

        total_weight = (
            self.transmission_efficiency_weight
            + self.reflection_efficiency_weight
            + self.reflection_selectivity_weight
        )
        if total_weight == 0:
            raise ValueError("at least one selective mirror weight must be nonzero")

        object.__setattr__(self, "transmissive_envelope", transmissive_envelope)
        object.__setattr__(self, "reflective_envelope", reflective_envelope)
        object.__setattr__(
            self,
            "transmission_efficiency_weight",
            self.transmission_efficiency_weight / total_weight,
        )
        object.__setattr__(
            self,
            "reflection_efficiency_weight",
            self.reflection_efficiency_weight / total_weight,
        )
        object.__setattr__(
            self,
            "reflection_selectivity_weight",
            self.reflection_selectivity_weight / total_weight,
        )

    def evaluate(self, spectrum: OpticalSpectrum) -> float:
        return self.evaluate_components(spectrum)["selective_mirror_fom"]

    def gradient(
        self,
        spectrum: OpticalSpectrum,
        spectrum_gradient: OpticalSpectrumGradient,
    ) -> NDArray:
        return self.gradient_components(spectrum, spectrum_gradient)[
            "selective_mirror_fom_gradient"
        ]

    def evaluate_components(self, spectrum: OpticalSpectrum) -> dict[str, float]:
        self._validate_spectrum(spectrum)
        wavelengths = spectrum.wavelengths
        transmissivity = spectrum.transmissivity
        reflectivity = spectrum.reflectivity

        useful_transmission = np.trapezoid(
            self.transmissive_envelope * transmissivity, wavelengths
        )
        useful_reflection = np.trapezoid(
            self.reflective_envelope * reflectivity, wavelengths
        )

        transmission_denom = np.trapezoid(self.transmissive_envelope, wavelengths)
        reflection_denom = np.trapezoid(reflectivity, wavelengths)
        reflection_selectivity_denom = np.trapezoid(
            self.reflective_envelope, wavelengths
        )

        if transmission_denom == 0.0:
            transmission_efficiency = 0.0
        else:
            transmission_efficiency = useful_transmission / transmission_denom

        if reflection_denom == 0.0:
            reflection_efficiency = 0.0
        else:
            reflection_efficiency = useful_reflection / reflection_denom

        if reflection_selectivity_denom == 0.0:
            reflection_selectivity = 0.0
        else:
            reflection_selectivity = (
                useful_reflection / reflection_selectivity_denom
            )

        selective_mirror_fom = (
            self.transmission_efficiency_weight * transmission_efficiency
            + self.reflection_efficiency_weight * reflection_efficiency
            + self.reflection_selectivity_weight * reflection_selectivity
        )

        return {
            "transmission_efficiency": transmission_efficiency,
            "reflection_efficiency": reflection_efficiency,
            "reflection_selectivity": reflection_selectivity,
            "selective_mirror_fom": selective_mirror_fom,
        }

    def gradient_components(
        self,
        spectrum: OpticalSpectrum,
        spectrum_gradient: OpticalSpectrumGradient,
    ) -> dict[str, NDArray]:
        self._validate_spectrum(spectrum)
        wavelengths = spectrum.wavelengths
        reflectivity = spectrum.reflectivity

        if spectrum_gradient.dR_dd.shape[0] != len(wavelengths):
            raise ValueError("gradient arrays must use wavelengths as axis 0")
        if spectrum_gradient.dR_dd.ndim != 2:
            raise ValueError(
                "SelectiveMirrorObjective currently expects gradient arrays "
                "with shape (number_of_wavelengths, number_of_gradients)"
            )

        number_of_gradients = spectrum_gradient.dT_dd.shape[-1]
        transmission_efficiency_gradient = np.zeros(number_of_gradients)
        reflection_efficiency_gradient = np.zeros(number_of_gradients)
        reflection_selectivity_gradient = np.zeros(number_of_gradients)

        eta_T_denom = np.trapezoid(self.transmissive_envelope, wavelengths)
        sel_R_denom = np.trapezoid(self.reflective_envelope, wavelengths)
        f_l = np.trapezoid(
            self.reflective_envelope * reflectivity,
            wavelengths,
        )
        g_l = np.trapezoid(reflectivity, wavelengths)

        for gradient_index in range(number_of_gradients):
            dT_dd = spectrum_gradient.dT_dd[:, gradient_index]
            dR_dd = spectrum_gradient.dR_dd[:, gradient_index]

            if eta_T_denom == 0.0:
                transmission_efficiency_gradient[gradient_index] = 0.0
            else:
                transmission_efficiency_gradient[gradient_index] = (
                    np.trapezoid(
                        self.transmissive_envelope * dT_dd,
                        wavelengths,
                    )
                    / eta_T_denom
                )

            if sel_R_denom == 0.0:
                reflection_selectivity_gradient[gradient_index] = 0.0
            else:
                reflection_selectivity_gradient[gradient_index] = (
                    np.trapezoid(
                        self.reflective_envelope * dR_dd,
                        wavelengths,
                    )
                    / sel_R_denom
                )

            if g_l == 0.0:
                reflection_efficiency_gradient[gradient_index] = 0.0
            else:
                gp_l = np.trapezoid(dR_dd, wavelengths)
                fp_l = np.trapezoid(self.reflective_envelope * dR_dd, wavelengths)
                reflection_efficiency_gradient[gradient_index] = (
                    g_l * fp_l - f_l * gp_l
                ) / g_l**2

        selective_mirror_fom_gradient = (
            self.transmission_efficiency_weight * transmission_efficiency_gradient
            + self.reflection_efficiency_weight * reflection_efficiency_gradient
            + self.reflection_selectivity_weight * reflection_selectivity_gradient
        )

        return {
            "transmission_efficiency_gradient": transmission_efficiency_gradient,
            "reflection_efficiency_gradient": reflection_efficiency_gradient,
            "reflection_selectivity_gradient": reflection_selectivity_gradient,
            "selective_mirror_fom_gradient": selective_mirror_fom_gradient,
        }

    def _validate_spectrum(self, spectrum: OpticalSpectrum) -> None:
        if spectrum.reflectivity.ndim != 1 or spectrum.transmissivity.ndim != 1:
            raise ValueError(
                "SelectiveMirrorObjective currently expects one-dimensional spectra"
            )
        if spectrum.reflectivity.shape != self.reflective_envelope.shape:
            raise ValueError("spectrum arrays and envelopes must have matching shape")
