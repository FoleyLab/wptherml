import numpy as np
import pytest

import wptherml
from wptherml.objectives import SelectiveMirrorObjective
from wptherml.solvers import TMMSolver
from wptherml.spectra import OpticalSpectrum, OpticalSpectrumGradient
from wptherml.structures import MultilayerStructure


def _driver_args():
    return {
        "wavelength_list": [500e-9, 504e-9, 5],
        "material_list": ["Air", "TiO2", "SiO2", "Ag", "Air"],
        "thickness_list": [0, 200e-9, 100e-9, 5e-9, 0],
    }


def _legacy_driver(args=None):
    return wptherml.SpectrumFactory().spectrum_factory("Tmm", args or _driver_args())


def _structure_from_driver(driver):
    return MultilayerStructure(
        materials=driver.material_array,
        thicknesses=driver.thickness_array,
        wavelengths=driver.wavelength_array,
        angles=np.array([driver.incident_angle]),
        refractive_indices=driver._refractive_index_array,
    )


def test_multilayer_structure_validation():
    structure = MultilayerStructure(
        materials=["Air", "SiO2", "Air"],
        thicknesses=np.array([0.0, 100e-9, 0.0]),
        wavelengths=np.array([500e-9, 600e-9]),
    )

    assert structure.number_of_layers == 3
    assert structure.number_of_wavelengths == 2

    with pytest.raises(ValueError, match="same length"):
        MultilayerStructure(
            materials=["Air", "SiO2"],
            thicknesses=np.array([0.0, 100e-9, 0.0]),
            wavelengths=np.array([500e-9]),
        )


def test_optical_spectrum_creation():
    wavelengths = np.array([500e-9, 600e-9])
    reflectivity = np.array([0.1, 0.2])
    transmissivity = np.array([0.7, 0.6])
    emissivity = 1 - reflectivity - transmissivity

    spectrum = OpticalSpectrum(
        wavelengths=wavelengths,
        reflectivity=reflectivity,
        transmissivity=transmissivity,
        emissivity=emissivity,
    )

    assert np.allclose(spectrum.emissivity, [0.2, 0.2])

    with pytest.raises(ValueError, match="shapes must match"):
        OpticalSpectrum(
            wavelengths=wavelengths,
            reflectivity=reflectivity,
            transmissivity=transmissivity[:1],
            emissivity=emissivity,
        )


def test_tmm_solver_vectorized_backend_matches_current_driver():
    driver = _legacy_driver()
    structure = _structure_from_driver(driver)

    spectrum = TMMSolver(backend="vectorized").solve(structure, polarizations="p")

    assert np.allclose(spectrum.reflectivity, driver.reflectivity_array)
    assert np.allclose(spectrum.transmissivity, driver.transmissivity_array)
    assert np.allclose(spectrum.emissivity, driver.emissivity_array)


def test_tmm_solver_serial_backend_matches_current_driver():
    driver = _legacy_driver()
    structure = _structure_from_driver(driver)

    spectrum = TMMSolver(backend="serial").solve(structure, polarizations="p")

    assert np.allclose(spectrum.reflectivity, driver.reflectivity_array)
    assert np.allclose(spectrum.transmissivity, driver.transmissivity_array)
    assert np.allclose(spectrum.emissivity, driver.emissivity_array)


def test_tmm_solver_serial_and_vectorized_agree():
    driver = _legacy_driver()
    structure = _structure_from_driver(driver)

    serial = TMMSolver(backend="serial").solve(structure, polarizations="p")
    vectorized = TMMSolver(backend="vectorized").solve(structure, polarizations="p")

    assert np.allclose(serial.reflectivity, vectorized.reflectivity)
    assert np.allclose(serial.transmissivity, vectorized.transmissivity)
    assert np.allclose(serial.emissivity, vectorized.emissivity)


def test_selective_mirror_objective_values_match_legacy_method():
    args = {
        "wavelength_list": [300e-9, 6000e-9, 1000],
        "material_list": ["Air", "SiO2", "Al2O3", "Air"],
        "thickness_list": [0, 500e-9, 500e-9, 0],
        "reflective_window_wn": [2000, 2400],
        "transmissive_window_nm": [350, 700],
        "transmission_efficiency_weight": 0.6,
        "reflection_efficiency_weight": 0.4,
        "reflection_selectivity_weight": 0.0,
    }
    driver = _legacy_driver(args)
    driver.transmissivity_array = np.copy(driver.transmissive_envelope)
    driver.reflectivity_array = np.copy(driver.reflective_envelope)
    driver.emissivity_array = 1 - driver.reflectivity_array - driver.transmissivity_array
    driver.compute_selective_mirror_fom()

    objective = SelectiveMirrorObjective(
        driver.transmissive_envelope,
        driver.reflective_envelope,
        0.6,
        0.4,
        0.0,
    )
    components = objective.evaluate_components(driver.spectrum)

    assert np.isclose(
        components["transmission_efficiency"], driver.transmission_efficiency
    )
    assert np.isclose(
        components["reflection_efficiency"], driver.reflection_efficiency
    )
    assert np.isclose(components["selective_mirror_fom"], driver.selective_mirror_fom)


def test_selective_mirror_objective_gradients_match_legacy_method():
    args = {
        "wavelength_list": [300e-9, 6000e-9, 180],
        "material_list": ["Air", "SiO2", "Al2O3", "SiO2", "Air"],
        "thickness_list": [0, 600e-9, 700e-9, 600e-9, 0],
        "reflective_window_wn": [2000, 2400],
        "transmissive_window_nm": [350, 700],
        "gradient_list": [1, 2, 3],
    }
    driver = _legacy_driver(args)
    driver.compute_selective_mirror_fom()
    driver.compute_selective_mirror_fom_gradient()

    objective = SelectiveMirrorObjective(
        driver.transmissive_envelope,
        driver.reflective_envelope,
        driver.transmission_efficiency_weight,
        driver.reflection_efficiency_weight,
        driver.reflection_selectivity_weight,
    )
    gradient = OpticalSpectrumGradient(
        dR_dd=driver.reflectivity_gradient_array,
        dT_dd=driver.transmissivity_gradient_array,
        dE_dd=driver.emissivity_gradient_array,
    )
    components = objective.gradient_components(driver.spectrum, gradient)

    assert np.allclose(
        components["transmission_efficiency_gradient"],
        driver.transmission_efficiency_gradient,
    )
    assert np.allclose(
        components["reflection_efficiency_gradient"],
        driver.reflection_efficiency_gradient,
    )
    assert np.allclose(
        components["selective_mirror_fom_gradient"],
        driver.selective_mirror_fom_gradient,
    )
