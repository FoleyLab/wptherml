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



def test_compute_spectrum_gradient():

    # simple structure - just glass layer
    test_args_1 = {
        "wavelength_list": [600e-9, 602e-9, 3],
        "material_list": [
            "Air",
            "SiO2",
            "Air",
        ],
        "thickness_list": [0, 200e-9, 0],
    }
    
    # 2-layer structure - just glass coated gold
    test_args_2 = {
        "wavelength_list": [600e-9, 602e-9, 3],
        "material_list": [
            "Air",
            "SiO2",
            "Au",
            "Air",
        ],
        "thickness_list": [0, 200e-9, 10e-9, 0],
    }
    
    # create instance of class
    ts1 = sf.spectrum_factory("Tmm", test_args_1)
    # hard-code RI using data from old WPTherml package
    ts1._refractive_index_array[0, :] = np.array([1+0j, 1.47779002+0.j, 1+0j])
    # run compute_spectrum method so that new RI values get incorporated 
    # into kz and kx arrays
    ts1.compute_spectrum()
    # compute spectrum gradient
    ts1.compute_spectrum_gradient()

    expected_T_prime_1 = 230498.36605705
    expected_R_prime_1 = -230498.36605705

    assert np.isclose(ts1.transmissivity_gradient_array[0,0], expected_T_prime_1)
    assert np.isclose(ts1.reflectivity_gradient_array[0,0], expected_R_prime_1)

        
    # create instance of class
    ts2 = sf.spectrum_factory("Tmm", test_args_2)
    # hard-code RI using data from old WPTherml package
    ts2._refractive_index_array[0, :] = np.array([1+0j, 1.47779002+0.j, 0.24463382+3.085112j, 1+0j])
    # run compute_spectrum method so that new RI values get incorporated 
    # into kz and kx arrays
    ts2.compute_spectrum()
    # compute spectrum gradient
    ts2.compute_spectrum_gradient()

    expected_R_prime_2 = np.array([-3782692.48735643, 32391843.05104597])
    expected_T_prime_2 = np.array([3248937.92608693, -37935758.73941045])
    expected_EPS_prime_2 = np.array([533754.5612695, 5543915.68836448])

    assert np.allclose(ts2.transmissivity_gradient_array[0,:], expected_T_prime_2)
    assert np.allclose(ts2.reflectivity_gradient_array[0,:], expected_R_prime_2)
    assert np.allclose(ts2.emissivity_gradient_array[0,:], expected_EPS_prime_2 )

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
