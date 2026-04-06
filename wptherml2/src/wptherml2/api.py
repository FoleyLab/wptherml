"""Public API for transfer-matrix calculations."""

from __future__ import annotations

from .tmm.gradients import compute_thickness_gradients
from .tmm.solve import solve_tmm
from .types import GradientResult, SpectrumResult, TmmRequest


def compute_observables(request: TmmRequest) -> SpectrumResult:
    """Compute spectral observables for a transfer-matrix request."""

    return solve_tmm(request)


def compute_gradients(request: TmmRequest) -> GradientResult:
    """Compute thickness gradients for spectral observables."""

    return compute_thickness_gradients(request)
