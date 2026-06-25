"""Numerical guards for the transfer-matrix kernels.

These tests are deliberately independent of the legacy driver so they keep
protecting correctness as the kernels are rewritten for performance:

* ``test_matches_frozen_baseline`` pins R/T/E and gradients to a snapshot
  captured from the known-good implementation (see ``_make_baseline.py``).
* ``test_gradients_match_finite_difference`` checks the analytic thickness
  gradients against central finite differences -- an external ground truth
  that does not depend on either the snapshot or the legacy code.
"""

import os

import numpy as np
import pytest

from wptherml.solvers import TMMSolver
from wptherml.structures import MultilayerStructure


REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "_baseline_reference.npz")


def _load_reference():
    if not os.path.exists(REFERENCE_PATH):
        pytest.skip("baseline reference snapshot not present")
    return np.load(REFERENCE_PATH)


def _structure_from_reference(ref, thicknesses=None):
    return MultilayerStructure(
        materials=[f"layer_{i}" for i in range(ref["refractive_indices"].shape[1])],
        thicknesses=ref["thicknesses"] if thicknesses is None else thicknesses,
        wavelengths=ref["wavelengths"],
        angles=ref["angles"],
        refractive_indices=ref["refractive_indices"],
    )


def test_matches_frozen_baseline():
    ref = _load_reference()
    structure = _structure_from_reference(ref)

    for backend in ("vectorized", "serial"):
        spectrum = TMMSolver(backend=backend).solve(structure, polarizations=["s", "p"])
        assert np.allclose(spectrum.reflectivity, ref[f"R_{backend}"], atol=1e-12)
        assert np.allclose(spectrum.transmissivity, ref[f"T_{backend}"], atol=1e-12)
        assert np.allclose(spectrum.emissivity, ref[f"E_{backend}"], atol=1e-12)

    gradient = TMMSolver(backend="vectorized").solve_gradients(
        structure, ref["gradient_layers"], polarizations=["s", "p"]
    )
    # Gradients have magnitude ~1/length (~1e7); the adjoint kernel differs from
    # the snapshot only at relative machine precision, so compare with rtol.
    assert np.allclose(gradient.dR_dd, ref["dR_dd"], rtol=1e-6, atol=1e-6)
    assert np.allclose(gradient.dT_dd, ref["dT_dd"], rtol=1e-6, atol=1e-6)
    assert np.allclose(gradient.dE_dd, ref["dE_dd"], rtol=1e-6, atol=1e-6)


def test_serial_and_vectorized_agree_multi_angle_multi_pol():
    ref = _load_reference()
    structure = _structure_from_reference(ref)

    serial = TMMSolver(backend="serial").solve(structure, polarizations=["s", "p"])
    vectorized = TMMSolver(backend="vectorized").solve(structure, polarizations=["s", "p"])

    assert np.allclose(serial.reflectivity, vectorized.reflectivity)
    assert np.allclose(serial.transmissivity, vectorized.transmissivity)
    assert np.allclose(serial.emissivity, vectorized.emissivity)


def test_gradients_match_finite_difference():
    ref = _load_reference()
    gradient_layers = ref["gradient_layers"]
    base_thicknesses = ref["thicknesses"].astype(float)

    structure = _structure_from_reference(ref)
    analytic = TMMSolver(backend="vectorized").solve_gradients(
        structure, gradient_layers, polarizations=["s", "p"]
    )

    step = 1e-12  # 1 pm finite-difference step on layer thicknesses
    for grad_index, layer in enumerate(gradient_layers):
        forward = base_thicknesses.copy()
        backward = base_thicknesses.copy()
        forward[layer] += step
        backward[layer] -= step

        spectrum_forward = TMMSolver(backend="vectorized").solve(
            _structure_from_reference(ref, forward), polarizations=["s", "p"]
        )
        spectrum_backward = TMMSolver(backend="vectorized").solve(
            _structure_from_reference(ref, backward), polarizations=["s", "p"]
        )

        dR_fd = (spectrum_forward.reflectivity - spectrum_backward.reflectivity) / (2 * step)
        dT_fd = (spectrum_forward.transmissivity - spectrum_backward.transmissivity) / (2 * step)

        assert np.allclose(analytic.dR_dd[..., grad_index], dR_fd, atol=1e-3, rtol=1e-4)
        assert np.allclose(analytic.dT_dd[..., grad_index], dT_fd, atol=1e-3, rtol=1e-4)
