"""Tests for the standalone, driver-free material lookup path."""

import contextlib
import io

import numpy as np

import wptherml
from wptherml.materials import build_refractive_index_array
from wptherml.structures import MultilayerStructure


def _legacy_refractive_indices(materials, thicknesses, wavelength_list):
    args = {
        "material_list": materials,
        "thickness_list": thicknesses,
        "wavelength_list": wavelength_list,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        driver = wptherml.SpectrumFactory().spectrum_factory("Tmm", args)
    return driver._refractive_index_array, driver.wavelength_array


def test_build_refractive_index_array_matches_legacy():
    materials = ["Air", "SiO2", "TiO2", "Ag", "Air"]
    thicknesses = [0, 200e-9, 100e-9, 10e-9, 0]
    wavelength_list = [400e-9, 1200e-9, 60]

    legacy_ri, wavelengths = _legacy_refractive_indices(
        materials, thicknesses, wavelength_list
    )
    new_ri = build_refractive_index_array(materials, wavelengths)

    assert new_ri.shape == legacy_ri.shape
    assert np.allclose(new_ri, legacy_ri)


def test_from_spec_matches_legacy_bridge_spectrum():
    materials = ["Air", "SiO2", "Au", "Air"]
    thicknesses = np.array([0, 200e-9, 10e-9, 0])
    wavelength_list = [400e-9, 800e-9, 100]

    legacy_ri, wavelengths = _legacy_refractive_indices(
        materials, thicknesses, wavelength_list
    )

    structure = MultilayerStructure.from_spec(
        materials=materials,
        thicknesses=thicknesses,
        wavelengths=wavelengths,
    )

    assert np.allclose(structure.refractive_indices, legacy_ri)

    spectrum = wptherml.TMMSolver().solve(structure, polarizations="p")
    assert spectrum.reflectivity.shape == wavelengths.shape
    assert np.all(np.isfinite(spectrum.reflectivity))


def test_assign_material_handles_case_insensitivity():
    wavelengths = np.linspace(400e-9, 800e-9, 10)
    lower = build_refractive_index_array(["air", "sio2", "air"], wavelengths)
    upper = build_refractive_index_array(["Air", "SiO2", "Air"], wavelengths)
    assert np.allclose(lower, upper)
