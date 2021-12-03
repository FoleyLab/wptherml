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


def test_pm_grad():
    """
    structure = {
        'Material_List' : ['Air', 'SiO2', 'Air'],
        ### Thicknesses just chosen arbitrarily, replace with "optimal" values
        'Thickness_List': [0, 200e-9, 0],
        ### add a number to Gradient_List to optimize over more layers
        'Gradient_List': [1],
        'Lambda_List': [600e-9, 602e-9, 3],
        }
    """

    test_args = {
        "wavelength_list": [600e-9, 602e-9, 3],
        "material_list": [
            "Air",
            "SiO2",
            "Air",
        ],
        "thickness_list": [0, 200e-9, 0],
    }

    ts = sf.spectrum_factory("Tmm", test_args)
    _kz = 15475380.92450645+0.j
    _phil = 3.09507618+0.j
    pml = ts._compute_pm_analytical_gradient(_kz, _phil)

    
    print(pml)

    expected_pml = np.array([[-719600.49694137+15458641.26899191j, 0+0j],
    [0.00000000e+00 +0j, -719600.49694137-15458641.26899191j]])

    assert np.allclose(pml, expected_pml)



def test_tm_grad():
    """
    structure = {
        'Material_List' : ['Air', 'SiO2', 'Air'],
        ### Thicknesses just chosen arbitrarily, replace with "optimal" values
        'Thickness_List': [0, 200e-9, 0],
        ### add a number to Gradient_List to optimize over more layers
        'Gradient_List': [1],
        'Lambda_List': [600e-9, 602e-9, 3],
        }
    """

    test_args = {
        "wavelength_list": [600e-9, 602e-9, 3],
        "material_list": [
            "Air",
            "SiO2",
	    "Air",
        ],
        "thickness_list": [0, 200e-9, 0],
    }
    # create instance of class
    ts = sf.spectrum_factory("Tmm", test_args)

    # define the z-component of wavevector exactly matching a air/sio2/air @ \lambda=600 nm
    # case from original wptherml release
    _kz = np.array([10471975.51196598+0.j, 15475380.92450645+0.j, 10471975.51196598+0.j])

    # k0 = 2 * pi / \lambda
    _k0 = 10471975.511965977

    #  thickness array 200 nm glass
    _d = np.array([0, 2e-07, 0])

    # _kz * d for glass layer
    _phil = 3.09507618+0.j

    # refractive index array for air/glass/air @ \lambda = 600 nm
    # exactly matching original wptherml release case
    _ri = np.array([1+0j, 1.47779002+0.j, 1+0j])

    # compute the gradient of the transfer matrix
    M, theta, ctheta = ts._compute_tm_gradient(_ri, _k0, _kz, _d, 1)

    # this is the expected gradient of the transfer matrix from original wptherml release
    expected_M0 = np.array([[-7.19600497e+05+16652636.92188255j, -5.82076609e-11 +6191988.90249864j],
    [0.00000000e+00 -6191988.90249864j, -7.19600497e+05-16652636.92188255j]])

    assert np.allclose(M, expected_M0)
