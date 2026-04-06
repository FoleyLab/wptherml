"""Minimal material-model protocol for future extension."""

from __future__ import annotations

from typing import Protocol

from ..types import ArrayC, ArrayF


class MaterialModel(Protocol):
    """Protocol for wavelength-dependent refractive-index providers."""

    def refractive_index(self, wavelength_m: ArrayF) -> ArrayC:
        """Return complex refractive index over a wavelength grid."""
