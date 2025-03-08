"""
Unit and regression test for the wpspec package.
"""

# Import package, test suite, and other packages as needed
import wptherml
import numpy as np
import pytest
import sys

sf = wptherml.SpectrumFactory()


def test_insert_layer():
    test_1_args = {
        "wavelength_list": [501e-9, 501e-9, 1],
        "material_list": [
            "Air",
            "TiO2",
            "SiO2",
            "Air",
        ],
        "thickness_list": [0, 200e-9, 100e-9, 0],
    }

    test_2_args = {
        "wavelength_list": [501e-9, 501e-9, 1],
        "material_list": [
            "Air",
            "TiO2",
            "SiO2",
            "Ag",
            "Air",
        ],
        "thickness_list": [0, 200e-9, 100e-9, 5e-9, 0],
    }

    test1 = sf.spectrum_factory("Tmm", test_1_args)
    test2 = sf.spectrum_factory("Tmm", test_2_args)

    test1.insert_layer(3, 5e-9)
    test1.material_Ag(3)
    test1.compute_spectrum()

    assert np.allclose(test1.reflectivity_array, test2.reflectivity_array)
    assert np.allclose(test1.transmissivity_array, test2.transmissivity_array)
    assert np.allclose(test1.emissivity_array, test2.emissivity_array)


def test_remove_layer():
    test_1_args = {
        "wavelength_list": [501e-9, 501e-9, 1],
        "material_list": [
            "Air",
            "TiO2",
            "SiO2",
            "Air",
        ],
        "thickness_list": [0, 200e-9, 100e-9, 0],
    }

    test_2_args = {
        "wavelength_list": [501e-9, 501e-9, 1],
        "material_list": [
            "Air",
            "TiO2",
            "SiO2",
            "Ag",
            "Air",
        ],
        "thickness_list": [0, 200e-9, 100e-9, 5e-9, 0],
    }

    test1 = sf.spectrum_factory("Tmm", test_1_args)
    test2 = sf.spectrum_factory("Tmm", test_2_args)

    test2.remove_layer(3)
    test2.compute_spectrum()

    assert np.allclose(test1.reflectivity_array, test2.reflectivity_array)
    assert np.allclose(test1.transmissivity_array, test2.transmissivity_array)
    assert np.allclose(test1.emissivity_array, test2.emissivity_array)


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


def test_compute_explicit_angle_spectrum():
    """test against a simple 230 nm Ag slab at lambda = 501 nm"""
    args = {
        "wavelength_list": [500e-9, 502e-9, 3],
        "material_list": ["Air", "Ag", "Air"],
        "thickness_list": [0, 230e-9, 0],
        "Temperature": 500,
        "therml": True,
    }

    sf = wptherml.SpectrumFactory()
    test = sf.spectrum_factory("Tmm", args)

    # hard-code in the RI values of Ag
    # 500 nm
    test._refractive_index_array[0, 1] = 0.04998114 + 3.13267845j
    # 501 nm
    test._refractive_index_array[1, 1] = 0.04997774 + 3.14242568j
    # 502 nm
    test._refractive_index_array[2, 1] = 0.04997488 + 3.15210009j

    # compute explicit-angle spectrum
    test.compute_explicit_angle_spectrum()

    # expected results from wptherml calculations with this Ag slab
    _expected_r_p = np.array(
        [
            0.98177278,
            0.98137611,
            0.97953671,
            0.97496043,
            0.96789884,
            0.96979929,
            0.991839,
        ]
    )
    _expected_r_s = np.array(
        [
            0.98180453,
            0.98219551,
            0.98388276,
            0.98740085,
            0.99207765,
            0.9964686,
            0.99930033,
        ]
    )
    _expected_t_p = np.array(
        [
            1.75624700e-08,
            1.75520517e-08,
            1.77152966e-08,
            1.92085994e-08,
            2.30502233e-08,
            1.72502686e-08,
            1.21003041e-09,
        ]
    )
    _expected_t_s = np.array(
        [
            1.75126607e-08,
            1.63066111e-08,
            1.19085368e-08,
            5.89754096e-09,
            1.89387786e-09,
            3.36368599e-10,
            1.28525918e-11,
        ]
    )
    _expected_e_p = np.array(
        [
            0.0182272,
            0.01862387,
            0.02046328,
            0.02503955,
            0.03210114,
            0.03020069,
            0.008161,
        ]
    )
    _expected_e_s = np.array(
        [
            0.01819546,
            0.01780447,
            0.01611723,
            0.01259915,
            0.00792235,
            0.0035314,
            0.00069967,
        ]
    )

    assert np.allclose(test.reflectivity_array_p[:, 1], _expected_r_p, 5e-3)
    assert np.allclose(test.reflectivity_array_s[:, 1], _expected_r_s, 5e-3)
    assert np.allclose(test.transmissivity_array_p[:, 1], _expected_t_p, 5e-3)
    assert np.allclose(test.transmissivity_array_s[:, 1], _expected_t_s, 5e-3)
    assert np.allclose(test.emissivity_array_p[:, 1], _expected_e_p, 5e-3)
    assert np.allclose(test.emissivity_array_s[:, 1], _expected_e_s, 5e-3)


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
    _kz = 15475380.92450645 + 0.0j
    _phil = 3.09507618 + 0.0j
    pml = ts._compute_pm_analytical_gradient(_kz, _phil)

    print(pml)

    expected_pml = np.array(
        [
            [-719600.49694137 + 15458641.26899191j, 0 + 0j],
            [0.00000000e00 + 0j, -719600.49694137 - 15458641.26899191j],
        ]
    )

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
    ts1._refractive_index_array[0, :] = np.array([1 + 0j, 1.47779002 + 0.0j, 1 + 0j])
    # run compute_spectrum method so that new RI values get incorporated
    # into kz and kx arrays
    ts1.compute_spectrum()
    # compute spectrum gradient
    ts1.compute_spectrum_gradient()

    expected_T_prime_1 = 230498.36605705
    expected_R_prime_1 = -230498.36605705

    assert np.isclose(ts1.transmissivity_gradient_array[0, 0], expected_T_prime_1)
    assert np.isclose(ts1.reflectivity_gradient_array[0, 0], expected_R_prime_1)

    # create instance of class
    ts2 = sf.spectrum_factory("Tmm", test_args_2)
    # hard-code RI using data from old WPTherml package
    ts2._refractive_index_array[0, :] = np.array(
        [1 + 0j, 1.47779002 + 0.0j, 0.24463382 + 3.085112j, 1 + 0j]
    )
    # run compute_spectrum method so that new RI values get incorporated
    # into kz and kx arrays
    ts2.compute_spectrum()
    # compute spectrum gradient
    ts2.compute_spectrum_gradient()

    expected_R_prime_2 = np.array([-3782692.48735643, 32391843.05104597])
    expected_T_prime_2 = np.array([3248937.92608693, -37935758.73941045])
    expected_EPS_prime_2 = np.array([533754.5612695, 5543915.68836448])

    assert np.allclose(ts2.transmissivity_gradient_array[0, :], expected_T_prime_2)
    assert np.allclose(ts2.reflectivity_gradient_array[0, :], expected_R_prime_2)
    assert np.allclose(ts2.emissivity_gradient_array[0, :], expected_EPS_prime_2)


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
    _kz = np.array(
        [10471975.51196598 + 0.0j, 15475380.92450645 + 0.0j, 10471975.51196598 + 0.0j]
    )

    # k0 = 2 * pi / \lambda
    _k0 = 10471975.511965977

    #  thickness array 200 nm glass
    _d = np.array([0, 2e-07, 0])

    # _kz * d for glass layer
    _phil = 3.09507618 + 0.0j

    # refractive index array for air/glass/air @ \lambda = 600 nm
    # exactly matching original wptherml release case
    _ri = np.array([1 + 0j, 1.47779002 + 0.0j, 1 + 0j])

    # compute the gradient of the transfer matrix
    M, theta, ctheta = ts._compute_tm_gradient(_ri, _k0, _kz, _d, 1)

    # this is the expected gradient of the transfer matrix from original wptherml release
    expected_M0 = np.array(
        [
            [-7.19600497e05 + 16652636.92188255j, -5.82076609e-11 + 6191988.90249864j],
            [0.00000000e00 - 6191988.90249864j, -7.19600497e05 - 16652636.92188255j],
        ]
    )

    assert np.allclose(M, expected_M0)

    
def test_selective_mirror_fom():
    """
    Test the computation of the selective mirror figure of merit against the following cases:

    1. The transmission exactly matches the transmissive_envelope, the reflectivity is zero everywhere
       and the weights are 0.6, 0.4 for transmission_efficiency and reflection_efficiency, respectively

    2. The transmission is zero everywhere, the reflectivity exactly matches the reflective_envelope
       and the weights are 0.25, 0.75 for transmission_efficiency and reflection_efficiency, respectively

    3. The transmission and reflectivity exactly match the windows and the weights are 0.5/0.5
    """
    # guess thickness for glass - totally arbitrary
    d1 = 500e-9

    # guess thickness for Al2O3 - totally arbitrary
    d2 = 500e-9

    # set weights for T and R for each test
    T_weight_1 = 0.6
    T_weight_2 = 0.25
    T_weight_3 = 0.5

    R_weight_1 = 0.4
    R_weight_2 = 0.75
    R_weight_3 = 0.5

    test_args1 = {
        "wavelength_list": [300e-9, 6000e-9, 1000],
        "Material_List": ["Air", "SiO2", "Al2O3", "SiO2", "Al2O3", "Air"],
        "Thickness_List": [0, d1, d2, d1, d2, 0],
        "reflective_window_wn": [2000, 2400],
        "transmissive_window_nm": [350, 700],
        "transmission_efficiency_weight": T_weight_1,
        "reflection_efficiency_weight": R_weight_1,
        "reflection_selectivity_weight" : 0,
    }
    test_args2 = {
        "wavelength_list": [300e-9, 6000e-9, 1000],
        "Material_List": ["Air", "SiO2", "Al2O3", "SiO2", "Al2O3", "Air"],
        "Thickness_List": [0, d1, d2, d1, d2, 0],
        "reflective_window_wn": [2000, 2400],
        "transmissive_window_nm": [350, 700],
        "transmission_efficiency_weight": T_weight_2,
        "reflection_efficiency_weight": R_weight_2,
        "reflection_selectivity_weight" : 0,
    }
    test_args3 = {
        "wavelength_list": [300e-9, 6000e-9, 1000],
        "Material_List": ["Air", "SiO2", "Al2O3", "SiO2", "Al2O3", "Air"],
        "Thickness_List": [0, d1, d2, d1, d2, 0],
        "reflective_window_wn": [2000, 2400],
        "transmissive_window_nm": [350, 700],
        "transmission_efficiency_weight": T_weight_3,
        "reflection_efficiency_weight": R_weight_3,
        "reflection_selectivity_weight" : 0,
    }

    # instantiate the factory
    sf = wptherml.SpectrumFactory()

    # create 3 test instances
    test1 = sf.spectrum_factory("Tmm", test_args1)
    test2 = sf.spectrum_factory("Tmm", test_args2)
    test3 = sf.spectrum_factory("Tmm", test_args3)

    # set T spectra for each test case
    test1.transmissivity_array = np.copy(test1.transmissive_envelope)  # <== perfect T
    test2.transmissivity_array = np.copy(
        np.zeros_like(test2.wavelength_array)
    )  # <== zeros everywhere
    test3.transmissivity_array = np.copy(test2.transmissive_envelope)  # <== perfect T

    # set R spectra for each test case
    test1.reflectivity_array = np.copy(
        np.zeros_like(test1.wavelength_array)
    )  # <== zeros everywhere
    test2.reflectivity_array = np.copy(test2.reflective_envelope)  # <== perfect R
    test3.reflectivity_array = np.copy(test2.reflective_envelope)  # <== perfect R

    # compute foms for each case
    test1.compute_selective_mirror_fom()
    test2.compute_selective_mirror_fom()
    test3.compute_selective_mirror_fom()

    # set the expected values
    # test 1
    _norm_1 = 1 / (T_weight_1 + R_weight_1)
    _expected_T_fom_1 = 1.0
    _expected_R_fom_1 = 0.0
    _expected_SM_fom_1 = (T_weight_1 / _norm_1) * _expected_T_fom_1 + (
        R_weight_1 / _norm_1
    ) * _expected_R_fom_1

    # test 2
    _norm_2 = 1 / (T_weight_2 + R_weight_2)
    _expected_T_fom_2 = 0.0
    _expected_R_fom_2 = 1.0
    _expected_SM_fom_2 = (T_weight_2 / _norm_2) * _expected_T_fom_2 + (
        R_weight_2 / _norm_2
    ) * _expected_R_fom_2

    # test 3
    _norm_3 = 1 / (T_weight_3 + R_weight_3)
    _expected_T_fom_3 = 1.0
    _expected_R_fom_3 = 1.0
    _expected_SM_fom_3 = (T_weight_3 / _norm_3) * _expected_T_fom_3 + (
        R_weight_3 / _norm_3
    ) * _expected_R_fom_3

    # check the weights
    assert np.isclose(test1.transmission_efficiency_weight, T_weight_1 / _norm_1)
    assert np.isclose(test2.transmission_efficiency_weight, T_weight_2 / _norm_2)
    assert np.isclose(test3.transmission_efficiency_weight, T_weight_3 / _norm_3)

    assert np.isclose(test1.reflection_efficiency_weight, R_weight_1 / _norm_1)
    assert np.isclose(test2.reflection_efficiency_weight, R_weight_2 / _norm_2)
    assert np.isclose(test3.reflection_efficiency_weight, R_weight_3 / _norm_3)

    # compare expected to computed values for all 9 foms
    assert np.isclose(test1.transmission_efficiency, _expected_T_fom_1)
    assert np.isclose(test2.transmission_efficiency, _expected_T_fom_2)
    assert np.isclose(test3.transmission_efficiency, _expected_T_fom_3)

    assert np.isclose(test1.reflection_efficiency, _expected_R_fom_1)
    assert np.isclose(test2.reflection_efficiency, _expected_R_fom_2)
    assert np.isclose(test3.reflection_efficiency, _expected_R_fom_3)

    assert np.isclose(test1.selective_mirror_fom, _expected_SM_fom_1)
    assert np.isclose(test2.selective_mirror_fom, _expected_SM_fom_2)
    assert np.isclose(test3.selective_mirror_fom, _expected_SM_fom_3)


def test_selective_mirror_fom_gradient():
    """
    Test the gradient of the selective mirror foms against finite differences for
    layer 4 for a small DBR structure
    """

    # guess thickness for glass
    d1 = 5000e-9 / (4 * 1.5)

    # guess thickness for Al2O3
    d2 = 5000e-9 / (4 * 1.85)

    test_args = {
        "wavelength_list": [300e-9, 6000e-9, 1000],
        "Material_List": [
            "Air",
            "SiO2",
            "Al2O3",
            "SiO2",
            "Al2O3",
            "SiO2",
            "Al2O3",
            "SiO2",
            "Al2O3",
            "SiO2",
            "Al2O3",
            "Air",
        ],
        "Thickness_List": [0, d1, d2, d1, d2, d1, d2, d1, d2, d1, d2, 0],
        "reflective_window_wn": [2000, 2400],
        "transmissive_window_nm": [350, 700],
    }

    sf = wptherml.SpectrumFactory()

    # create an instance of the DBR called test
    test = sf.spectrum_factory("Tmm", test_args)
    # compute FOMs
    test.compute_selective_mirror_fom()
    # compute FOM gradients
    test.compute_selective_mirror_fom_gradient()

    # select layer corresponding to the gradient element to check
    _n_layer = 4  # <== layer 4

    # set step size for finite differences
    dx = 0.01e-9

    # now do numerical gradients
    test.thickness_array[_n_layer] += dx
    test.compute_spectrum()
    test.compute_selective_mirror_fom()

    # store the foms from the forward-displaced structure
    _r_fom_f = test.reflection_efficiency
    _t_fom_f = test.transmission_efficiency

    # move to backwards-displaced structure
    test.thickness_array[_n_layer] -= 2 * dx
    test.compute_spectrum()
    test.compute_selective_mirror_fom()

    # store the foms from the backward-displaced structure
    _r_fom_b = test.reflection_efficiency
    _t_fom_b = test.transmission_efficiency

    # compute centered-finite-difference (cfd) approximation to the gradient elements for _n_layer
    _r_grad = (_r_fom_f - _r_fom_b) / (2 * dx)
    _t_grad = (_t_fom_f - _t_fom_b) / (2 * dx)

    # compare cfd to analytic gradient
    np.isclose(_r_grad, test.reflection_efficiency_gradient[_n_layer])
    np.isclose(_r_grad, test.transmission_efficiency_gradient[_n_layer])
