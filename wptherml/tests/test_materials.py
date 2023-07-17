"""
Unit tests for the Materials class
"""

# Import package, test suite, and other packages as needed
import wptherml
import numpy as np
import pytest
import sys

material_test = wptherml.Materials()


def test_material_from_file():
    """tests material_from_file method using filenames for the SiO2 and TiN files,
    specifically "SiO2_ir.txt" using tabulated n and k at lambda=636 nm of 6.359999999E-7 1.45693 0.00000
    and "TiN_ellipsometry_data.txt" using tabulated n and k at lambda=1106 nm of 1.106906906906907e-06 2.175019337515494 5.175973473259225
    """

    _expected_sio2_n = 1.45693
    _expected_sio2_k = 0.00000

    _expected_tin_n = 2.175019337515494
    _expected_tin_k = 5.175973473259225

    # instance for SiO2 test
    material_test._create_test_multilayer(central_wavelength=636e-9)
    material_test.material_from_file(1, "SiO2_ir.txt")

    _result_sio2_n = np.real(material_test._refractive_index_array[1, 1])
    _result_sio2_k = np.imag(material_test._refractive_index_array[1, 1])

    # instance for TiN test
    material_test._create_test_multilayer(central_wavelength=1.106906906906907e-06)
    # define central layer as TiN
    material_test.material_from_file(1, "TiN_ellipsometry_data.txt")

    _result_tin_n = np.real(material_test._refractive_index_array[1, 1])
    _result_tin_k = np.imag(material_test._refractive_index_array[1, 1])

    assert np.isclose(_result_sio2_n, _expected_sio2_n, 1e-3)
    assert np.isclose(_result_sio2_k, _expected_sio2_k, 1e-3)
    assert np.isclose(_result_tin_n, _expected_tin_n, 1e-3)
    assert np.isclose(_result_tin_k, _expected_tin_k, 1e-3)


def test_material_sio2():
    """tests material_sio2 method using tabulated n and k at lambda=636 nm
    6.359999999E-7 1.45693 0.00000"""

    expected_n = 1.45693
    expected_k = 0.00000

    # create test multilayer that has 3 layers and wavelength array centered at 636 nm
    material_test._create_test_multilayer(central_wavelength=636e-9)
    # define central layer as siO2
    material_test.material_SiO2(1)

    result_n = np.real(material_test._refractive_index_array[1, 1])
    result_k = np.imag(material_test._refractive_index_array[1, 1])

    assert np.isclose(result_n, expected_n, 1e-3)
    assert np.isclose(result_k, expected_k, 1e-3)

def test_material_sio2_udm():
    """tests material_sio2_udm method using tabulated n and k at en = 0.01207 eV:
       1.207814e-02	2.046290e+00	3.283908e-02
    """

    expected_n = 2.046290e+00
    expected_k = 3.283908e-02

    central_energy_ev = 1.207814e-02
    central_wavelength_nm = 1.2398e+3 / central_energy_ev 
    central_wavelength_m = central_wavelength_nm * 1e-9

    # create test multilayer that has 3 layers and wavelength array centered at 0.01207 eV
    material_test._create_test_multilayer(central_wavelength=central_wavelength_m)
    # define central layer as siO2
    material_test.material_SiO2_UDM(1)

    result_n = np.real(material_test._refractive_index_array[1, 1])
    result_k = np.imag(material_test._refractive_index_array[1, 1])

    assert np.isclose(result_n, expected_n, 1e-3)
    assert np.isclose(result_k, expected_k, 1e-3)

def test_material_si3n4():
    """tests material_si3n4 method using tabulated n and k at lambda = 3.738 microns
       0.000003738	1.913747161	0
    """

    expected_n = 1.913747161
    expected_k = 0.0

    central_wavelength_m = 0.000003738

    # create test multilayer that has 3 layers and wavelength array centered at 3.3738 microns
    material_test._create_test_multilayer(central_wavelength=central_wavelength_m)
    # define central layer as si3n4
    material_test.material_Si3N4(1)

    result_n = np.real(material_test._refractive_index_array[1, 1])
    result_k = np.imag(material_test._refractive_index_array[1, 1])

    assert np.isclose(result_n, expected_n, 1e-3)
    assert np.isclose(result_k, expected_k, 1e-3)

def test_material_zro2():
    """tests material_zro2 method using tabulated n and k at lambda = 695.2 nm 
       6.952E-07	2.144155168	0
    """

    expected_n = 2.144155168
    expected_k = 0.0

    central_wavelength_m = 6.952E-07

    # create test multilayer that has 3 layers and wavelength array centered at 695.2 nm
    material_test._create_test_multilayer(central_wavelength=central_wavelength_m)
    # define central layer as Zr02
    material_test.material_ZrO2(1)

    result_n = np.real(material_test._refractive_index_array[1, 1])
    result_k = np.imag(material_test._refractive_index_array[1, 1])

    assert np.isclose(result_n, expected_n, 1e-3)
    assert np.isclose(result_k, expected_k, 1e-3)

def test_material_h2o():
    """tests material_H2O method using the fact that n = 1.33 + 0j for all lambda"""

    expected_n = 1.33
    expected_k = 0.00000

    # create test multilayer that has 3 layers and wavelength array centered at 636 nm
    material_test._create_test_multilayer(central_wavelength=636e-9)
    # define central layer as H20
    material_test.material_H2O(1)

    result_n = np.real(material_test._refractive_index_array[1, 1])
    result_k = np.imag(material_test._refractive_index_array[1, 1])

    assert np.isclose(result_n, expected_n, 1e-3)
    assert np.isclose(result_k, expected_k, 1e-3)


def test_material_tio2():
    """tests material_tio2 method using tabulated n and k at lambda=664 nm
    6.639999999E-7 2.377021563 6.79e-10"""

    expected_n = 2.377021563
    expected_k = 6.79e-10

    # create test multilayer that has 3 layers and wavelength array centered at 664 nm
    material_test._create_test_multilayer(central_wavelength=664e-9)
    # define central layer as tiO2
    material_test.material_TiO2(1)

    result_n = np.real(material_test._refractive_index_array[1, 1])
    result_k = np.imag(material_test._refractive_index_array[1, 1])

    assert np.isclose(result_n, expected_n, 1e-3)
    assert np.isclose(result_k, expected_k, 1e-3)


def test_material_ta2o5():
    """Dictionaries from material_Ta2O5"""
    data1 = {
        "file": "data/Ta2O5_Rodriguez.txt",
        "lower_wavelength": 2.9494e-08,
        "upper_wavelength": 1.5143e-06,
        "test_wavelength": 3.6899e-08,
        "test_n": 8.6165e-01,
        "test_k": 2.9300e-01,
    }
    data2 = {
        "file": "data/Ta2O5_Bright.txt",
        "lower_wavelength": 5.0000e-07,
        "upper_wavelength": 1.0000e-03,
        "test_wavelength": 2.5907e-06,
        "test_n": 2.0246e00,
        "test_k": 7.5989e-03,
    }

    expected_n_1 = data1["test_n"]
    expected_k_1 = data1["test_k"]
    wavelength_1 = data1["test_wavelength"]

    expected_n_2 = data2["test_n"]
    expected_k_2 = data2["test_k"]
    wavelength_2 = data2["test_wavelength"]

    # create test multilayer for data1
    material_test._create_test_multilayer(central_wavelength=wavelength_1)
    # define central layer as Ta2O5 using data1
    material_test.material_Ta2O5(1)

    result_n_1 = np.real(material_test._refractive_index_array[1, 1])
    result_k_1 = np.imag(material_test._refractive_index_array[1, 1])

    # update test multilayer for data2
    material_test._create_test_multilayer(central_wavelength=wavelength_2)
    # define central layer as Ta2O5 using data2
    material_test.material_Ta2O5(1)

    result_n_2 = np.real(material_test._refractive_index_array[1, 1])
    result_k_2 = np.imag(material_test._refractive_index_array[1, 1])

    assert np.isclose(result_n_1, expected_n_1, 1e-3)
    assert np.isclose(result_k_1, expected_k_1, 1e-3)
    assert np.isclose(result_n_2, expected_n_2, 1e-3)
    assert np.isclose(result_k_2, expected_k_2, 1e-3)


def test_material_tin():
    """tests material_TiN method using tabulated n and k at lambda=1106 nm
    1.106906906906907e-06 2.175019337515494 5.175973473259225"""

    expected_n = 2.175019337515494
    expected_k = 5.175973473259225

    # create test multilayer that has 3 layers and wavelength array centered at 1106 nm
    material_test._create_test_multilayer(central_wavelength=1.106906906906907e-06)
    # define central layer as TiN
    material_test.material_TiN(1)

    result_n = np.real(material_test._refractive_index_array[1, 1])
    result_k = np.imag(material_test._refractive_index_array[1, 1])

    assert np.isclose(result_n, expected_n, 1e-3)
    assert np.isclose(result_k, expected_k, 1e-3)


def test_material_al():
    """tests material_Al method using tabulated n and k at lambda=206.64 nm
    2.06640E-07	1.26770E-01	2.35630E+00"""

    expected_n = 1.26770e-01
    expected_k = 2.35630e00

    # create test multilayer that has 3 layers and wavelength array centered at 206.64 nm
    material_test._create_test_multilayer(central_wavelength=2.06640e-07)

    # define central layer as Al
    material_test.material_Al(1)

    result_n = np.real(material_test._refractive_index_array[1, 1])
    result_k = np.imag(material_test._refractive_index_array[1, 1])

    assert np.isclose(result_n, expected_n, 1e-3)
    assert np.isclose(result_k, expected_k, 1e-3)


def test_material_hfo2():
    """tests material_HfO2 method using tabulated n and k at lambda=1082.00 nm
    1.082000E-06 1.880787E+00 0.000000E+00"""

    expected_n = 1.880586
    expected_k = 0.000000

    # create test multilayer that has 3 layers and wavelength array centered at 1082.0 nm
    material_test._create_test_multilayer(central_wavelength=1.082000e-06)

    # define central layer as HfO2
    material_test.material_HfO2(1)

    result_n = np.real(material_test._refractive_index_array[1, 1])
    result_k = np.imag(material_test._refractive_index_array[1, 1])

    assert np.isclose(result_n, expected_n, 1e-3)
    assert np.isclose(result_k, expected_k, 1e-3)


def test_material_au():
    """Dictionaries from material_Au"""
    data1 = {
        "file": "data/Au_JC_RI_f.txt",
        "lower_wavelength": 2e-07,
        "upper_wavelength": 1.00025e-06,
        "test_wavelength": 5.00188e-07,
        "test_n": 0.962208410850276,
        "test_k": 1.8695066263351445,
    }
    data2 = {
        "file": "data/Au_IR.txt",
        "lower_wavelength": 3.000000e-07,
        "upper_wavelength": 2.493000e-05,
        "test_wavelength": 5.024000e-06,
        "test_n": 3.031000e00,
        "test_k": 3.447000e01,
    }

    expected_n_1 = data1["test_n"]
    expected_k_1 = data1["test_k"]
    wavelength_1 = data1["test_wavelength"]

    expected_n_2 = data2["test_n"]
    expected_k_2 = data2["test_k"]
    wavelength_2 = data2["test_wavelength"]

    # create test multilayer for data1
    material_test._create_test_multilayer(central_wavelength=wavelength_1)
    # define central layer as au using data1
    material_test.material_Au(1)

    result_n_1 = np.real(material_test._refractive_index_array[1, 1])
    result_k_1 = np.imag(material_test._refractive_index_array[1, 1])

    # update test multilayer for data2
    material_test._create_test_multilayer(central_wavelength=wavelength_2)
    # define central layer as au using data2
    material_test.material_Au(1)

    result_n_2 = np.real(material_test._refractive_index_array[1, 1])
    result_k_2 = np.imag(material_test._refractive_index_array[1, 1])

    assert np.isclose(result_n_1, expected_n_1, 1e-3)
    assert np.isclose(result_k_1, expected_k_1, 1e-3)
    assert np.isclose(result_n_2, expected_n_2, 1e-3)
    assert np.isclose(result_k_2, expected_k_2, 1e-3)


def test_material_pt():
    """tests material_Pt method using tabulated n and k at lambda=610 nm
    6.1096e-06 5.4685e+00 2.4477e+01"""

    expected_n = 5.4685e00
    expected_k = 2.4477e01
    # create test multilayer that has 3 layers and wavelength array centered at 610 nm
    material_test._create_test_multilayer(central_wavelength=6.1096e-06)
    # define central layer as Pt
    material_test.material_Pt(1)

    result_n = np.real(material_test._refractive_index_array[1, 1])
    result_k = np.imag(material_test._refractive_index_array[1, 1])

    assert np.isclose(result_n, expected_n, 1e-3)
    assert np.isclose(result_k, expected_k, 1e-3)


def test_material_al2o3():
    """tests material_al2o3 method using tabulated n and k at lambda=500 nm
    5.00E-07 1.74007 0"""

    expected_n = 1.74007
    expected_k = 0

    # create test multilayer that has 3 layers and wavelength array centered at 500 nm
    material_test._create_test_multilayer(central_wavelength=5.00e-07)
    # define central layer as Al2O3
    material_test.material_Al2O3(1)

    result_n = np.real(material_test._refractive_index_array[1, 1])
    result_k = np.imag(material_test._refractive_index_array[1, 1])

    assert np.isclose(result_n, expected_n, 1e-3)
    assert np.isclose(result_k, expected_k, 1e-3)

def test_material_al2o3_udm():
    """tests material_al2o3_udm method using tabulated n and k at en = 0.01455 eV:
       1.455459e-02	2.498235e+00	6.013373e-03
    """

    expected_n = 2.498235e+00
    expected_k = 6.013373e-03

    central_energy_ev = 1.455459e-02
    central_wavelength_nm = 1.2398e+3 / central_energy_ev 
    central_wavelength_m = central_wavelength_nm * 1e-9

    # create test multilayer that has 3 layers and wavelength array centered at 636 nm
    material_test._create_test_multilayer(central_wavelength=central_wavelength_m)
    # define central layer as Al203
    material_test.material_Al2O3_UDM(1)

    result_n = np.real(material_test._refractive_index_array[1, 1])
    result_k = np.imag(material_test._refractive_index_array[1, 1])

    assert np.isclose(result_n, expected_n, 1e-3)
    assert np.isclose(result_k, expected_k, 1e-3)


def test_material_polystyrene():
    """tests material_Polystyrene method using tabulated n and k at lambda=500 nm
    0.0000005	1.60021	6.11E-07"""

    expected_n = 1.60021
    expected_k = 6.11e-07

    # create test multilayer that has 3 layers and wavelength array centered at 500 nm
    material_test._create_test_multilayer(central_wavelength=0.0000005)
    # define central layer as polystyrene
    material_test.material_polystyrene(1)

    result_n = np.real(material_test._refractive_index_array[1, 1])
    result_k = np.imag(material_test._refractive_index_array[1, 1])

    assert np.isclose(result_n, expected_n, 1e-3)
    assert np.isclose(result_k, expected_k, 1e-3)


def test_material_rh():
    """tests material_Rh method using tabulated n and k at lambda=564 nm
    5.636E-07	2	5.11"""

    expected_n = 2
    expected_k = 5.11
    # create test multilayer that has 3 layers and wavelength array centered at 564 nm
    material_test._create_test_multilayer(central_wavelength=5.636e-07)
    # define central layer as Rh
    material_test.material_Rh(1)

    result_n = np.real(material_test._refractive_index_array[1, 1])
    result_k = np.imag(material_test._refractive_index_array[1, 1])

    assert np.isclose(result_n, expected_n, 1e-3)
    assert np.isclose(result_k, expected_k, 1e-3)


def test_material_ru():
    """tests material_Ru method using tabulated n and k at lambda=750 nm
    0.0000007508  5.1101514677  4.1107371518"""

    expected_n = 5.1101514677
    expected_k = 4.1107371518
    # create test multilayer that has 3 layers and wavelength array centered at 750 nm
    material_test._create_test_multilayer(central_wavelength=0.0000007508)
    # define central layer as Ru
    material_test.material_Ru(1)

    result_n = np.real(material_test._refractive_index_array[1, 1])
    result_k = np.imag(material_test._refractive_index_array[1, 1])

    assert np.isclose(result_n, expected_n, 1e-3)
    assert np.isclose(result_k, expected_k, 1e-3)


def test_material_aln():
    """Dictionaries from material_AlN"""
    data1 = {
        "file": "data/AlN_Pastrnak.txt",
        "lower_wavelength": 0.000000220,
        "upper_wavelength": 0.000005000,
        "test_wavelength": 0.000001082,
        "test_n": 2.129298842,
        "test_k": 0.000000000,
    }
    data2 = {
        "file": "data/AlN_Kischkat.txt",
        "lower_wavelength": 1.53846e-06,
        "upper_wavelength": 1.42857e-05,
        "test_wavelength": 6.26566e-06,
        "test_n": 1.87617,
        "test_k": 0.00264,
    }

    expected_n_1 = data1["test_n"]
    expected_k_1 = data1["test_k"]
    wavelength_1 = data1["test_wavelength"]

    expected_n_2 = data2["test_n"]
    expected_k_2 = data2["test_k"]
    wavelength_2 = data2["test_wavelength"]

    # create test multilayer for data1
    material_test._create_test_multilayer(central_wavelength=wavelength_1)
    # define central layer as AlN using data1
    material_test.material_AlN(1)

    result_n_1 = np.real(material_test._refractive_index_array[1, 1])
    result_k_1 = np.imag(material_test._refractive_index_array[1, 1])
    print(result_n_1, expected_n_1)
    # update test multilayer for data2
    material_test._create_test_multilayer(central_wavelength=wavelength_2)
    # define central layer as AlN using data2
    material_test.material_AlN(1)

    result_n_2 = np.real(material_test._refractive_index_array[1, 1])
    result_k_2 = np.imag(material_test._refractive_index_array[1, 1])

    assert np.isclose(result_n_1, expected_n_1, 1e-3)
    assert np.isclose(result_k_1, expected_k_1, 1e-3)
    assert np.isclose(result_n_2, expected_n_2, 1e-3)
    assert np.isclose(result_k_2, expected_k_2, 1e-3)


def test_material_w():
    """Dictionaries from material_W"""
    data1 = {
        "file": "data/W_Rakic.txt",
        "lower_wavelength": 1.53846e-06,
        "upper_wavelength": 1.2398e-05,
        "test_wavelength": 1.3620e-06,
        "test_n": 3.1024e00,
        "test_k": 4.3555e00,
    }

    data2 = {
        "file": "data/W_Ordal.txt",
        "lower_wavelength": 6.67000e-07,
        "upper_wavelength": 2.00000e-04,
        "test_wavelength": 2.22000e-05,
        "test_n": 27.5991880,
        "test_k": 83.5173670,
    }

    expected_n_1 = data1["test_n"]
    expected_k_1 = data1["test_k"]
    wavelength_1 = data1["test_wavelength"]

    expected_n_2 = data2["test_n"]
    expected_k_2 = data2["test_k"]
    wavelength_2 = data2["test_wavelength"]

    # create test multilayer for data1
    material_test._create_test_multilayer(central_wavelength=wavelength_1)
    # define central layer as W using data1
    material_test.material_W(1)

    result_n_1 = np.real(material_test._refractive_index_array[1, 1])
    result_k_1 = np.imag(material_test._refractive_index_array[1, 1])

    # update test multilayer for data2
    material_test._create_test_multilayer(central_wavelength=wavelength_2)
    # define central layer as W using data2
    material_test.material_W(1)

    result_n_2 = np.real(material_test._refractive_index_array[1, 1])
    result_k_2 = np.imag(material_test._refractive_index_array[1, 1])

    assert np.isclose(result_n_1, expected_n_1, 1e-3)
    assert np.isclose(result_k_1, expected_k_1, 1e-3)
    assert np.isclose(result_n_2, expected_n_2, 1e-3)
    assert np.isclose(result_k_2, expected_k_2, 1e-3)


def test_material_si():
    """Dictionaries from material_Si"""
    data1 = {
        "file": "data/Si_Schinke.txt",
        "lower_wavelength": 0.000000250,
        "upper_wavelength": 0.000001450,
        "test_wavelength": 0.000000790,
        "test_n": 3.677000000,
        "test_k": 0.005688800,
    }

    data2 = {
        "file": "data/Si_Shkondin.txt",
        "lower_wavelength": 2.00000e-06,
        "upper_wavelength": 2.00000e-04,
        "test_wavelength": 0.00001052000,
        "test_n": 3.46969768200,
        "test_k": 0.00008077950,
    }

    expected_n_1 = data1["test_n"]
    expected_k_1 = data1["test_k"]
    wavelength_1 = data1["test_wavelength"]

    expected_n_2 = data2["test_n"]
    expected_k_2 = data2["test_k"]
    wavelength_2 = data2["test_wavelength"]

    # create test multilayer for data1
    material_test._create_test_multilayer(central_wavelength=wavelength_1)
    # define central layer as Si using data1
    material_test.material_Si(1)

    result_n_1 = np.real(material_test._refractive_index_array[1, 1])
    result_k_1 = np.imag(material_test._refractive_index_array[1, 1])

    # update test multilayer for data2
    material_test._create_test_multilayer(central_wavelength=wavelength_2)
    # define central layer as Si using data2
    material_test.material_Si(1)

    result_n_2 = np.real(material_test._refractive_index_array[1, 1])
    result_k_2 = np.imag(material_test._refractive_index_array[1, 1])

    assert np.isclose(result_n_1, expected_n_1, 1e-3)
    assert np.isclose(result_k_1, expected_k_1, 1e-3)
    assert np.isclose(result_n_2, expected_n_2, 1e-3)
    assert np.isclose(result_k_2, expected_k_2, 1e-3)


def test_material_re():
    """Dictionaries from material_Re"""
    data1 = {
        "file": "data/Re_Windt.txt",
        "lower_wavelength": 2.36e-09,
        "upper_wavelength": 1.2157e-07,
        "test_wavelength": 5.391e-08,
        "test_n": 0.786,
        "test_k": 0.723,
    }

    data2 = {
        "file": "data/Re_Palik.txt",
        "lower_wavelength": 0.0000004000,
        "upper_wavelength": 0.0000060000,
        "test_wavelength": 0.0000007508,
        "test_n": 3.3403575532,
        "test_k": 2.7994102694,
    }

    expected_n_1 = data1["test_n"]
    expected_k_1 = data1["test_k"]
    wavelength_1 = data1["test_wavelength"]

    expected_n_2 = data2["test_n"]
    expected_k_2 = data2["test_k"]
    wavelength_2 = data2["test_wavelength"]

    # create test multilayer for data1
    material_test._create_test_multilayer(central_wavelength=wavelength_1)
    # define central layer as Re using data1
    material_test.material_Re(1)

    result_n_1 = np.real(material_test._refractive_index_array[1, 1])
    result_k_1 = np.imag(material_test._refractive_index_array[1, 1])

    # update test multilayer for data2
    material_test._create_test_multilayer(central_wavelength=wavelength_2)
    # define central layer as Re using data2
    material_test.material_Re(1)

    result_n_2 = np.real(material_test._refractive_index_array[1, 1])
    result_k_2 = np.imag(material_test._refractive_index_array[1, 1])

    assert np.isclose(result_n_1, expected_n_1, 1e-3)
    assert np.isclose(result_k_1, expected_k_1, 1e-3)
    assert np.isclose(result_n_2, expected_n_2, 1e-3)
    assert np.isclose(result_k_2, expected_k_2, 1e-3)


def test_material_ag():
    """Dictionaries from material_Ag"""
    data1 = {
        "file": "data/Ag_JC.txt",
        "lower_wavelength": 1.87900e-07,
        "upper_wavelength": 1.93700e-06,
        "test_wavelength": 3.97400e-07,
        "test_n": 5.00000e-02,
        "test_k": 2.07000e00,
    }

    data2 = {
        "file": "data/Ag_Yang.txt",
        "lower_wavelength": 2.70000e-07,
        "upper_wavelength": 2.49200e-05,
        "test_wavelength": 2.08400e-06,
        "test_n": 4.61000e-01,
        "test_k": 1.47300e01,
    }

    expected_n_1 = data1["test_n"]
    expected_k_1 = data1["test_k"]
    wavelength_1 = data1["test_wavelength"]

    expected_n_2 = data2["test_n"]
    expected_k_2 = data2["test_k"]
    wavelength_2 = data2["test_wavelength"]

    # create test multilayer for data1
    material_test._create_test_multilayer(central_wavelength=wavelength_1)
    # define central layer as Ag using data1
    material_test.material_Ag(1)

    result_n_1 = np.real(material_test._refractive_index_array[1, 1])
    result_k_1 = np.imag(material_test._refractive_index_array[1, 1])

    # update test multilayer for data2
    material_test._create_test_multilayer(central_wavelength=wavelength_2)
    # define central layer as Ag using data2
    material_test.material_Ag(1)

    result_n_2 = np.real(material_test._refractive_index_array[1, 1])
    result_k_2 = np.imag(material_test._refractive_index_array[1, 1])

    assert np.isclose(result_n_1, expected_n_1, 1e-3)
    assert np.isclose(result_k_1, expected_k_1, 1e-3)
    assert np.isclose(result_n_2, expected_n_2, 1e-3)
    assert np.isclose(result_k_2, expected_k_2, 1e-3)


def test_material_pb():
    """Dictionaries from material_Pb"""
    data1 = {
        "file": "data/Pb_Werner.txt",
        "lower_wavelength": 1.758600000e-08,
        "upper_wavelength": 2.479684000e-06,
        "test_wavelength": 3.262700000e-08,
        "test_n": 9.903000000e-01,
        "test_k": 2.206000000e-01,
    }

    data2 = {
        "file": "data/Pb_Ordal.txt",
        "lower_wavelength": 0.00000066700000,
        "upper_wavelength": 0.00066700000000,
        "test_wavelength": 0.00000286000000,
        "test_n": 3.940000000,
        "test_k": 15.600000000,
    }

    expected_n_1 = data1["test_n"]
    expected_k_1 = data1["test_k"]
    wavelength_1 = data1["test_wavelength"]

    expected_n_2 = data2["test_n"]
    expected_k_2 = data2["test_k"]
    wavelength_2 = data2["test_wavelength"]

    # create test multilayer for data1
    material_test._create_test_multilayer(central_wavelength=wavelength_1)
    # define central layer as Pb using data1
    material_test.material_Pb(1)

    result_n_1 = np.real(material_test._refractive_index_array[1, 1])
    result_k_1 = np.imag(material_test._refractive_index_array[1, 1])

    # update test multilayer for data2
    material_test._create_test_multilayer(central_wavelength=wavelength_2)
    # define central layer as Pb using data2
    material_test.material_Pb(1)

    result_n_2 = np.real(material_test._refractive_index_array[1, 1])
    result_k_2 = np.imag(material_test._refractive_index_array[1, 1])

    assert np.isclose(result_n_1, expected_n_1, 1e-3)
    assert np.isclose(result_k_1, expected_k_1, 1e-3)
    assert np.isclose(result_n_2, expected_n_2, 1e-3)
    assert np.isclose(result_k_2, expected_k_2, 1e-3)


def test_read_AM():

    # create test multilayer for a wavelength array centered at 615 nm
    material_test._create_test_multilayer(central_wavelength=615e-9)

    # Now store the AM1.5 data centered at 615 nm to an array
    _AM_15_data = material_test._read_AM()

    # this is the expected value of the AM1.5 spectrum at 615 nm
    _expected_value = 1325400000

    # test to see if the expected value is close to the read value
    assert np.isclose(_AM_15_data[1], _expected_value, 1e-3)


def test_read_atmospheric_transmissivity():

    # create test multilayer for a wavelength array centered at 7.1034 microns
    material_test._create_test_multilayer(central_wavelength=7.1034e-6)

    # Now store the AM1.5 data centered at 615 nm to an array
    _atmospheric_transmissivity = material_test._read_Atmospheric_Transmissivity()

    # this is the expected value of the AM1.5 spectrum at 615 nm
    _expected_value = 0.561289

    # test to see if the expected value is close to the read value
    assert np.isclose(_atmospheric_transmissivity[1], _expected_value, 1e-3)
