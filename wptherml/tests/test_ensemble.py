"""Tests for the thickness-uncertainty ensemble sampler."""

import numpy as np
import pytest

import wptherml
from wptherml.ensemble import ThicknessEnsemble
from wptherml.structures import MultilayerStructure
from wptherml.tmm_fast import solve_rt


def _structure():
    return MultilayerStructure.from_spec(
        materials=["Air", "SiO2", "TiO2", "SiO2", "Air"],
        thicknesses=[0, 230e-9, 120e-9, 230e-9, 0],
        wavelengths=np.linspace(400e-9, 800e-9, 120),
    )


def test_sampling_only_perturbs_internal_layers():
    structure = _structure()
    ensemble = ThicknessEnsemble(
        structure, relative_sigma=0.1, number_of_samples=64, seed=0
    )
    thicknesses = ensemble.sample_thicknesses()

    assert thicknesses.shape == (64, structure.number_of_layers)
    # Terminal layers are never perturbed.
    assert np.all(thicknesses[:, 0] == structure.thicknesses[0])
    assert np.all(thicknesses[:, -1] == structure.thicknesses[-1])
    # Internal layers vary.
    assert np.std(thicknesses[:, 1]) > 0.0
    # Non-negativity is enforced.
    assert np.all(thicknesses >= 0.0)


def test_sigma_default_is_relative_to_nominal():
    structure = _structure()
    ensemble = ThicknessEnsemble(structure, number_of_samples=10)
    nominal = np.asarray(structure.thicknesses)[ensemble.sample_layers]
    assert np.allclose(ensemble.sigma, 0.05 * nominal)


def test_absolute_sigma_scalar_applies_to_all_layers():
    structure = _structure()
    ensemble = ThicknessEnsemble(structure, sigma=5e-9, number_of_samples=10)
    assert np.allclose(ensemble.sigma, 5e-9)


def test_ensemble_solution_matches_per_replica_single_solve():
    structure = _structure()
    ensemble = ThicknessEnsemble(
        structure, sigma=8e-9, number_of_samples=32, seed=42
    )
    result = ensemble.solve(polarizations="p")

    assert result.reflectivity.shape == (32, structure.number_of_wavelengths)

    # A few replicas must equal an independent single-structure solve.
    for replica in (0, 5, 17, 31):
        R, _T, _E = solve_rt(
            structure.refractive_indices,
            structure.wavelengths,
            structure.angles,
            result.thicknesses[replica],
            "p",
        )
        assert np.allclose(result.reflectivity[replica], R[:, 0, 0], atol=1e-12)


def test_summary_statistics_and_quantiles():
    structure = _structure()
    ensemble = ThicknessEnsemble(structure, sigma=5e-9, number_of_samples=200, seed=1)
    result = ensemble.solve(polarizations="p")

    assert result.reflectivity_mean.shape == (structure.number_of_wavelengths,)
    assert result.reflectivity_std.shape == (structure.number_of_wavelengths,)
    assert np.all(result.reflectivity_std >= 0.0)

    lo, _, _ = result.quantile(0.05)
    hi, _, _ = result.quantile(0.95)
    assert np.all(hi >= lo)


def test_ensemble_gradients_shape_and_values():
    structure = _structure()
    ensemble = ThicknessEnsemble(structure, sigma=5e-9, number_of_samples=16, seed=7)
    result = ensemble.solve(polarizations="p", gradients=True)

    grads = result.gradients
    assert grads is not None
    n_grad = ensemble.sample_layers.size
    assert grads["dR_dd"].shape == (16, structure.number_of_wavelengths, n_grad)
    assert np.all(np.isfinite(grads["dR_dd"]))


def test_zero_sigma_reproduces_nominal_spectrum():
    structure = _structure()
    ensemble = ThicknessEnsemble(structure, sigma=0.0, number_of_samples=4, seed=0)
    result = ensemble.solve(polarizations="p")

    nominal = wptherml.TMMSolver().solve(structure, polarizations="p")
    for replica in range(result.number_of_samples):
        assert np.allclose(result.reflectivity[replica], nominal.reflectivity, atol=1e-12)
