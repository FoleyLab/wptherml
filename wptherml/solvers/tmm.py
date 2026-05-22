"""Transfer-matrix solvers built on the shared numerical kernels."""

import numpy as np

from .base import GradientSolver, Solver
from ..spectra import OpticalSpectrum, OpticalSpectrumGradient
from ..structures import MultilayerStructure
from ..tmm_core import (
    compute_spectrum_from_transfer_matrix,
    compute_spectrum_gradients_from_transfer_matrix,
    compute_transfer_matrix,
    compute_transfer_matrix_gradients,
    compute_transfer_matrices,
)


class TMMSolver(Solver, GradientSolver):
    """Solve multilayer optical spectra with transfer-matrix kernels.

    Parameters
    ----------
    backend
        ``"vectorized"`` uses batched kernels and is the recommended default.
        ``"serial"`` loops over wavelength/angle/polarization while still using
        the same ``tmm_core`` numerical kernels.
    """

    _SUPPORTED_BACKENDS = {"vectorized", "serial"}

    def __init__(self, backend: str = "vectorized") -> None:
        backend = backend.lower()
        if backend not in self._SUPPORTED_BACKENDS:
            supported = ", ".join(sorted(self._SUPPORTED_BACKENDS))
            raise ValueError(f"backend must be one of {supported}")
        self.backend = backend

    def solve(
        self,
        structure: MultilayerStructure,
        polarizations: str | list[str] | tuple[str, ...] | None = None,
    ) -> OpticalSpectrum:
        self._validate_structure_for_solving(structure)
        polarization_list = self._normalize_polarizations(polarizations)

        if self.backend == "vectorized":
            reflectivity, transmissivity, emissivity = self._solve_vectorized(
                structure, polarization_list
            )
        else:
            reflectivity, transmissivity, emissivity = self._solve_serial(
                structure, polarization_list
            )

        reflectivity = self._squeeze_spectrum_axes(
            reflectivity, structure, polarization_list
        )
        transmissivity = self._squeeze_spectrum_axes(
            transmissivity, structure, polarization_list
        )
        emissivity = self._squeeze_spectrum_axes(
            emissivity, structure, polarization_list
        )

        return OpticalSpectrum(
            wavelengths=structure.wavelengths,
            reflectivity=reflectivity,
            transmissivity=transmissivity,
            emissivity=emissivity,
            angles=structure.angles,
            polarizations=polarization_list,
        )

    def solve_gradients(
        self,
        structure: MultilayerStructure,
        gradient_layers,
        polarizations: str | list[str] | tuple[str, ...] | None = None,
    ) -> OpticalSpectrumGradient:
        self._validate_structure_for_solving(structure)
        polarization_list = self._normalize_polarizations(polarizations)
        angles = self._angles(structure)
        n, k0, kz = self._wavevector_arrays(structure, angles)

        gradient_results = []
        for polarization in polarization_list:
            (
                transfer_matrix,
                transfer_matrix_gradient,
                _theta,
                cos_theta,
            ) = compute_transfer_matrix_gradients(
                n,
                k0,
                kz,
                structure.thicknesses,
                angles,
                polarization,
                gradient_layers,
            )
            gradient_results.append(
                compute_spectrum_gradients_from_transfer_matrix(
                    transfer_matrix,
                    transfer_matrix_gradient,
                    n,
                    cos_theta,
                )
            )

        dR_dd = np.stack([result[0] for result in gradient_results], axis=2)
        dT_dd = np.stack([result[1] for result in gradient_results], axis=2)
        dE_dd = np.stack([result[2] for result in gradient_results], axis=2)

        dR_dd = self._squeeze_gradient_axes(dR_dd, structure, polarization_list)
        dT_dd = self._squeeze_gradient_axes(dT_dd, structure, polarization_list)
        dE_dd = self._squeeze_gradient_axes(dE_dd, structure, polarization_list)

        return OpticalSpectrumGradient(dR_dd=dR_dd, dT_dd=dT_dd, dE_dd=dE_dd)

    def _solve_vectorized(
        self, structure: MultilayerStructure, polarizations: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        angles = self._angles(structure)
        n, k0, kz = self._wavevector_arrays(structure, angles)
        transfer_matrix, _theta, cos_theta = compute_transfer_matrices(
            n,
            k0,
            kz,
            structure.thicknesses,
            angles,
            polarizations,
        )
        return compute_spectrum_from_transfer_matrix(transfer_matrix, n, cos_theta)

    def _solve_serial(
        self, structure: MultilayerStructure, polarizations: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        wavelengths = structure.wavelengths
        angles = self._angles(structure)
        refractive_indices = structure.refractive_indices

        shape = (len(wavelengths), len(angles), len(polarizations))
        reflectivity = np.empty(shape, dtype=float)
        transmissivity = np.empty(shape, dtype=float)
        emissivity = np.empty(shape, dtype=float)

        for wavelength_index, wavelength in enumerate(wavelengths):
            k0 = 2 * np.pi / wavelength
            refractive_index = refractive_indices[wavelength_index, :]
            for angle_index, angle in enumerate(angles):
                kx = refractive_index[0] * np.sin(angle) * k0
                kz = np.sqrt((refractive_index * k0) ** 2 - kx**2)
                for polarization_index, polarization in enumerate(polarizations):
                    transfer_matrix, _theta, cos_theta = compute_transfer_matrix(
                        refractive_index,
                        k0,
                        kz,
                        structure.thicknesses,
                        angle,
                        polarization,
                    )
                    (
                        reflectivity[wavelength_index, angle_index, polarization_index],
                        transmissivity[
                            wavelength_index, angle_index, polarization_index
                        ],
                        emissivity[wavelength_index, angle_index, polarization_index],
                    ) = compute_spectrum_from_transfer_matrix(
                        transfer_matrix,
                        refractive_index,
                        cos_theta,
                    )

        return reflectivity, transmissivity, emissivity

    def _wavevector_arrays(
        self, structure: MultilayerStructure, angles: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        refractive_indices = structure.refractive_indices[:, np.newaxis, :]
        n = np.broadcast_to(
            refractive_indices,
            (
                structure.number_of_wavelengths,
                len(angles),
                structure.number_of_layers,
            ),
        )
        k0 = (2 * np.pi / structure.wavelengths)[:, np.newaxis]
        n0 = structure.refractive_indices[:, 0][:, np.newaxis]
        kx = n0 * np.sin(angles)[np.newaxis, :] * k0
        kz = np.sqrt((n * k0[:, :, np.newaxis]) ** 2 - kx[:, :, np.newaxis] ** 2)
        return n, k0, kz

    def _validate_structure_for_solving(self, structure: MultilayerStructure) -> None:
        if structure.refractive_indices is None:
            raise ValueError(
                "TMMSolver requires structure.refractive_indices to be provided"
            )

    def _angles(self, structure: MultilayerStructure) -> np.ndarray:
        if structure.angles is None:
            return np.array([0.0], dtype=float)
        return np.atleast_1d(np.asarray(structure.angles, dtype=float))

    def _normalize_polarizations(
        self, polarizations: str | list[str] | tuple[str, ...] | None
    ) -> list[str]:
        if polarizations is None:
            return ["p"]
        if isinstance(polarizations, str):
            polarization_list = [polarizations.lower()]
        else:
            polarization_list = [polarization.lower() for polarization in polarizations]

        invalid = [
            polarization
            for polarization in polarization_list
            if polarization not in {"s", "p"}
        ]
        if invalid:
            raise ValueError(f"polarizations must be 's' or 'p', got {invalid}")
        return polarization_list

    def _squeeze_spectrum_axes(
        self,
        array: np.ndarray,
        structure: MultilayerStructure,
        polarizations: list[str],
    ) -> np.ndarray:
        if len(self._angles(structure)) == 1 and len(polarizations) == 1:
            return array[:, 0, 0]
        return array

    def _squeeze_gradient_axes(
        self,
        array: np.ndarray,
        structure: MultilayerStructure,
        polarizations: list[str],
    ) -> np.ndarray:
        if len(self._angles(structure)) == 1 and len(polarizations) == 1:
            return array[:, 0, 0, :]
        return array
