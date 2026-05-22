"""Geometry and material definitions for stratified optical stacks."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class MultilayerStructure:
    """Material and geometry description for a multilayer stack.

    Angles are expressed in radians. Refractive indices are optional so that
    material lookup can remain a separate concern from geometry ownership.
    """

    materials: list[str]
    thicknesses: NDArray
    wavelengths: NDArray
    angles: NDArray | None = None
    refractive_indices: NDArray | None = None

    def __post_init__(self) -> None:
        self.materials = list(self.materials)
        self.thicknesses = np.asarray(self.thicknesses, dtype=float)
        self.wavelengths = np.asarray(self.wavelengths, dtype=float)
        if self.angles is not None:
            self.angles = np.atleast_1d(np.asarray(self.angles, dtype=float))
        if self.refractive_indices is not None:
            self.refractive_indices = np.asarray(
                self.refractive_indices, dtype=np.complex128
            )

        if self.thicknesses.ndim != 1:
            raise ValueError("thicknesses must be a one-dimensional array")
        if self.wavelengths.ndim != 1:
            raise ValueError("wavelengths must be a one-dimensional array")
        if self.angles is not None and self.angles.ndim != 1:
            raise ValueError("angles must be a one-dimensional array")
        if len(self.materials) != len(self.thicknesses):
            raise ValueError(
                "materials and thicknesses must have the same length "
                f"({len(self.materials)} != {len(self.thicknesses)})"
            )
        if self.refractive_indices is not None and self.refractive_indices.shape != (
            len(self.wavelengths),
            len(self.materials),
        ):
            raise ValueError(
                "refractive_indices must have shape "
                "(number_of_wavelengths, number_of_layers)"
            )

    @property
    def number_of_layers(self) -> int:
        return len(self.materials)

    @property
    def number_of_wavelengths(self) -> int:
        return len(self.wavelengths)
