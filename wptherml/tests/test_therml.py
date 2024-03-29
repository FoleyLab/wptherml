"""
Unit and regression test for the wpspec package.
"""

# Import package, test suite, and other packages as needed
import wptherml
import numpy as np
import pytest
import sys


def test_compute_power_density():
    test_args = {
        "wavelength_list": [100e-9, 30000e-9, 10000],
        "material_list": ["Air", "W", "Air"],
        "thickness_list": [0, 800e-9, 0],
        "temperature": 1500,
        "therml": True,
    }
    sf = wptherml.SpectrumFactory()
    test = sf.spectrum_factory("Tmm", test_args)

    """will test _compute_power_density method to
    see if the power density computed by integration
    of the blackbody spectrum is close to the prediction
    by the Stefan-Boltzmann law, where the latter should be exact
    """
    assert np.isclose(test.blackbody_power_density, test.stefan_boltzmann_law, 1e-2)


def test_compute_stpv_power_density():

    # define basic structure at 1500 K
    test_args = {
        "wavelength_list": [400e-9, 7000e-9, 1000],
        "material_list": ["Air", "TiN", "Air"],
        "thickness_list": [0, 400e-9, 0],
        "temperature": 5000,
        "therml": True,
    }
    sf = wptherml.SpectrumFactory()
    test = sf.spectrum_factory("Tmm", test_args)
    """will test _compute_stpv_power_density method to
    see if the power density computed by integration
    of the thermal emission spectrum of TiN at 5000 K is close to 
    the value that is generated by v1.0.0 of wptherml
    """
    assert np.isclose(test.stpv_power_density, 3863945.0, 1e4)


def test_compute_stpv_gradients():
    """unit test for the computation of the gradients of stpv quantities
    including the stpv_power_density and the spectral_efficiency"""
    test_args = {
        "wavelength_list": [400e-9, 7000e-9, 1000],
        "material_list": ["Air", "SiO2", "TiN", "Air"],
        "thickness_list": [0, 20e-9, 400e-9, 0],
        "temperature": 5000,
        "therml": True,
    }
    sf = wptherml.SpectrumFactory()
    test = sf.spectrum_factory("Tmm", test_args)
    test.compute_stpv_gradient()

    # get analytic gradients
    _analytic_stpv_power_density_gradient = test.stpv_power_density_gradient[0]
    _analytic_stpv_spectral_efficiency_gradient = (
        test.stpv_spectral_efficiency_gradient[0]
    )

    # define a displacement in thickness of SiO2
    _delta_d_sio2 = 0.1e-9
    # take forward step and store quantities
    test.thickness_array[1] += _delta_d_sio2
    test.compute_spectrum()
    test.compute_stpv()
    _stpv_power_density_f = test.stpv_power_density
    _stpv_spectral_efficiency_f = test.stpv_spectral_efficiency

    # take backward step and store quantities
    test.thickness_array[1] -= 2 * _delta_d_sio2
    test.compute_spectrum()
    test.compute_stpv()
    _stpv_power_density_b = test.stpv_power_density
    _stpv_spectral_efficiency_b = test.stpv_spectral_efficiency

    # compute gradients by centered fininte differences
    _numeric_stpv_power_density_gradient = (
        _stpv_power_density_f - _stpv_power_density_b
    ) / (2 * _delta_d_sio2)
    _numeric_stpv_spectral_efficiency_gradient = (
        _stpv_spectral_efficiency_f - _stpv_spectral_efficiency_b
    ) / (2 * _delta_d_sio2)

    # these gradients have big values so scale them relative to the numeric gradient
    _scaled_analytic_stpv_power_density_gradient = (
        _analytic_stpv_power_density_gradient / _numeric_stpv_power_density_gradient
    )

    # test stpv_power_density_gradient
    assert np.isclose(_scaled_analytic_stpv_power_density_gradient, 1, 2e-2)
    # test stpv_spectral_efficiency_gradient
    assert np.isclose(
        _numeric_stpv_spectral_efficiency_gradient,
        _analytic_stpv_spectral_efficiency_gradient,
        1e-2,
    )


def test_compute_stpv_efficiency():
    # define basic structure at 1500 K
    test_args = {
        "wavelength_list": [400e-9, 7000e-9, 1000],
        "material_list": ["Air", "TiN", "Air"],
        "thickness_list": [0, 400e-9, 0],
        "temperature": 1700,
        "therml": True,
    }
    sf = wptherml.SpectrumFactory()
    test = sf.spectrum_factory("Tmm", test_args)
    """will test _compute_stpv_power_density method to
    see if the power density computed by integration
    of the thermal emission spectrum of TiN at 5000 K is close to 
    the value that is generated by v1.0.0 of wptherml
    """
    assert np.isclose(test.stpv_spectral_efficiency, 0.3786094154907156, 5e-2)


def test_compute_luminous_efficiency():
    # define basic structure at 1500 K
    test_args = {
        "wavelength_list": [400e-9, 7000e-9, 1000],
        "material_list": ["Air", "TiN", "Air"],
        "thickness_list": [0, 400e-9, 0],
        "temperature": 5000,
        "therml": True,
    }
    sf = wptherml.SpectrumFactory()
    test = sf.spectrum_factory("Tmm", test_args)
    """will test _compute_stpv_power_density method to
    see if the power density computed by integration
    of the thermal emission spectrum of TiN at 5000 K is close to 
    the value that is generated by v1.0.0 of wptherml
    """
    assert np.isclose(test.luminous_efficiency, 0.20350729803724107, 1e-2)


def test_compute_cooling():
    test_args = {
        "wavelength_list": [300e-9, 60000e-9, 5000],
        "material_list": ["Air", "SiO2", "Air"],
        "thickness_list": [0, 230e-9, 0],
        "temperature": 300,
        "cooling": True,
    }
    sf = wptherml.SpectrumFactory()
    test = sf.spectrum_factory("Tmm", test_args)
    test._refractive_index_array[:, 1] = 2.4 + 0.2j

    _expected_radiative_cooling_power = 40.44509251986298
    _expected_atmospheric_warming_power = 21.973817620650525
    _expected_solar_warming_power = 426.9132402277394

    test.compute_cooling()

    test.compute_explicit_angle_spectrum()

    assert np.isclose(
        _expected_radiative_cooling_power, test.radiative_cooling_power, 1e-5
    )
    assert np.isclose(
        _expected_atmospheric_warming_power, test.atmospheric_warming_power, 1e-5
    )
    assert np.isclose(_expected_solar_warming_power, test.solar_warming_power, 1e-5)


def test_compute_cooling_gradient():
    """FINISH THIS UNIT TEST"""

    # define basic structure at 1500 K
    test_args = {
        "wavelength_list": [300e-9, 20000e-9, 5000],
        "material_list": ["Air", "TiN", "Air"],
        "thickness_list": [0, 230e-9, 0],
        "temperature": 300,
        "cooling": True,
    }
    sf = wptherml.SpectrumFactory()
    test = sf.spectrum_factory("Tmm", test_args)

    # get analytic gradient of the absorbed solar power
    test.compute_cooling_gradient()

    # define a displacement in thickness of SiO2
    _delta_d_sio2 = 1.0e-9

    test.incident_angle = test.solar_angle
    # get forward solar power
    test.thickness_array[1] += _delta_d_sio2
    test.compute_cooling()

    _solar_power_f = test.solar_warming_power
    _emitted_power_f = test.radiative_cooling_power
    _atmospheric_power_f = test.atmospheric_warming_power

    # get backward solar power
    test.thickness_array[1] -= 2 * _delta_d_sio2
    test.compute_cooling()
    _solar_power_b = test.solar_warming_power
    _emitted_power_b = test.radiative_cooling_power
    _atmospheric_power_b = test.atmospheric_warming_power

    _numeric_solar_warming_power_gradient = (_solar_power_f - _solar_power_b) / (
        2 * _delta_d_sio2
    )
    _numeric_radiative_cooling_power_gradient = (
        _emitted_power_f - _emitted_power_b
    ) / (2 * _delta_d_sio2)
    _numeric_atmospheric_warming_power_gradient = (
        _atmospheric_power_f - _atmospheric_power_b
    ) / (2 * _delta_d_sio2)

    # normalize the gradients by the numeric gradient
    _normalized_analytic_solar_power_gradient = (
        test.solar_warming_power_gradient / _numeric_solar_warming_power_gradient
    )
    _normalized_analytic_emitted_power_gradient = (
        test.radiative_cooling_power_gradient
        / _numeric_radiative_cooling_power_gradient
    )
    _normalized_analytic_atmospheric_power_gradient = (
        test.atmospheric_warming_power_gradient
        / _numeric_atmospheric_warming_power_gradient
    )

    # if the gradients are close, the normalized analytic gradient will be close to 1
    assert np.isclose(_normalized_analytic_solar_power_gradient[0], 1.0, 1e-3)
    assert np.isclose(_normalized_analytic_emitted_power_gradient[0], 1.0, 1e-3)
    assert np.isclose(_normalized_analytic_atmospheric_power_gradient[0], 1.0, 1e-3)
    assert np.isclose(
        test.atmospheric_warming_power_gradient[0],
        _numeric_atmospheric_warming_power_gradient,
        1e-2,
    )
