"""
Unit and regression test for the wpspec package.
"""

# Import package, test suite, and other packages as needed
import wpspecdev
import numpy as np
import pytest
import sys

sf = wpspecdev.SpectrumFactory()


def test_compute_spectrum():
    """tests public method in TmmDriver compute_spectrum()
    using several test cases with a structure as follows:

    'wavelength_list': [501e-9, 501e-9, 1],
    'material_list': ["Air", "TiO2", "SiO2", "Ag", "Au", "Pt", "AlN", "Al2O3", "Air"],
    'thickness_list': [0, 200e-9, 100e-9, 5e-9, 6e-9, 7e-9, 100e-9, 201e-9, 0]

    Under the following conditions:

    a. Normal incidence (independent of polarization)
    b. 45 degrees with s-polarilzed light
    c. 55 degrees with p-polarized light

    """

    expected_result_normal_incidence = np.array(
        [
            0.21399468,  # reflectance
            0.43566534,  # transmitttance
            0.35033999,  # emissivity
        ]
    )

    expected_result_45_degress_s_polarized = np.array(
        [
            0.09929336,  # reflectance
            0.33761702,  # transmittance
            0.56308962,  # emissivity
        ]
    )

    expected_result_55_degress_p_polarized = np.array(
        [
            0.28574605,  # reflectane
            0.37053201,  # transmittance
            0.34372194,  # emissivity
        ]
    )

    test_1_args = {
        "wavelength_list": [501e-9, 501e-9, 1],
        "material_list": [
            "Air",
            "TiO2",
            "SiO2",
            "Ag",
            "Au",
            "Pt",
            "AlN",
            "Al2O3",
            "Air",
        ],
        "thickness_list": [0, 200e-9, 100e-9, 5e-9, 6e-9, 7e-9, 100e-9, 201e-9, 0],
    }

    test_2_args = {
        "wavelength_list": [501e-9, 501e-9, 1],
        "material_list": [
            "Air",
            "TiO2",
            "SiO2",
            "Ag",
            "Au",
            "Pt",
            "AlN",
            "Al2O3",
            "Air",
        ],
        "thickness_list": [0, 200e-9, 100e-9, 5e-9, 6e-9, 7e-9, 100e-9, 201e-9, 0],
        "polarization": "s",
        "incident_angle": 45.0,
    }

    test_3_args = {
        "wavelength_list": [501e-9, 501e-9, 1],
        "material_list": [
            "Air",
            "TiO2",
            "SiO2",
            "Ag",
            "Au",
            "Pt",
            "AlN",
            "Al2O3",
            "Air",
        ],
        "thickness_list": [0, 200e-9, 100e-9, 5e-9, 6e-9, 7e-9, 100e-9, 201e-9, 0],
        "polarization": "p",
        "incident_angle": 55.0,
    }

    test1 = sf.spectrum_factory("Tmm", test_1_args)
    test2 = sf.spectrum_factory("Tmm", test_2_args)
    test3 = sf.spectrum_factory("Tmm", test_3_args)

    # test1
    assert np.isclose(
        test1.reflectivity_array[0], expected_result_normal_incidence[0], 5e-3
    )
    assert np.isclose(
        test1.transmissivity_array[0], expected_result_normal_incidence[1], 5e-3
    )
    assert np.isclose(
        test1.emissivity_array[0], expected_result_normal_incidence[2], 5e-3
    )

    # test2
    assert np.isclose(
        test2.reflectivity_array[0], expected_result_45_degress_s_polarized[0], 5e-3
    )
    assert np.isclose(
        test2.transmissivity_array[0], expected_result_45_degress_s_polarized[1], 5e-3
    )
    assert np.isclose(
        test2.emissivity_array[0], expected_result_45_degress_s_polarized[2], 5e-3
    )

    # test3
    assert np.isclose(
        test3.reflectivity_array[0], expected_result_55_degress_p_polarized[0], 5e-3
    )
    assert np.isclose(
        test3.transmissivity_array[0], expected_result_55_degress_p_polarized[1], 5e-3
    )
    assert np.isclose(
        test3.emissivity_array[0], expected_result_55_degress_p_polarized[2], 5e-3
    )
