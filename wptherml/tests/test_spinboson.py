"""
Unit and regression test for the wpspec package.
"""

# Import package, test suite, and other packages as needed
import wptherml
import numpy as np
import pytest
import sys
from numpy import linalg


def test_build_boson_basis():
    """
    Unit test for the build_boson_basis method.
    Tests against expected basis when the number of boson levels is 3 (basis includes |0>, |1>, |2>)
    """
    sf = wptherml.SpectrumFactory()
    args = {
        "number_of_boson_levels": 3,
    }

    test = sf.spectrum_factory("Spin-Boson", args)
    test.build_boson_basis()
    _expected_basis = np.matrix("1 0 0 ; 0 1 0 ; 0 0 1")
    assert np.allclose(_expected_basis, test.boson_basis)
    print(test.boson_basis)


def test_build_exciton_basis():
    """
    Unit test for the build_exciton_basis method.
    Tests against expected basis for three cases:

    (1) there is a single exciton
    (2) there are 2 excitons
    (3) there are 4 excitons
    """
    sf = wptherml.SpectrumFactory()

    # dictionaries for 3 cases
    args_1 = {
        "number_of_excitons": 1,
    }

    args_2 = {
        "number_of_excitons": 2,
    }

    args_3 = {
        "number_of_excitons": 4,
    }

    # instantiate 3 cases
    test_1 = sf.spectrum_factory("Spin-Boson", args_1)
    test_2 = sf.spectrum_factory("Spin-Boson", args_2)
    test_3 = sf.spectrum_factory("Spin-Boson", args_3)

    # build the exciton bases
    test_1.build_exciton_basis()
    test_2.build_exciton_basis()
    test_3.build_exciton_basis()

    _expected_one_exciton_basis = np.matrix("1 0 ; 0 1")
    _expected_two_exciton_basis = np.kron(
        _expected_one_exciton_basis, _expected_one_exciton_basis
    )
    _expected_four_exciton_basis = np.kron(
        _expected_one_exciton_basis,
        np.kron(_expected_one_exciton_basis, _expected_two_exciton_basis),
    )

    # for case 1 the single_exciton_basis should match the n_exciton_basis should match _expected_one_exciton_basis
    assert np.allclose(_expected_one_exciton_basis, test_1.single_exciton_basis)
    assert np.allclose(_expected_one_exciton_basis, test_1.n_exciton_basis)

    # for case 2, the single_exciton_basis should match the _expected_one_exciton_basis and the n_exciton_basis should match _expected_two_exciton_basis
    assert np.allclose(_expected_one_exciton_basis, test_2.single_exciton_basis)
    assert np.allclose(_expected_two_exciton_basis, test_2.n_exciton_basis)

    # for case 3, the single_exciton_basis should match the _expected_one_exciton_basis and the n_exciton_basis should match _expected_four_exciton_basis
    assert np.allclose(_expected_one_exciton_basis, test_3.single_exciton_basis)
    assert np.allclose(_expected_four_exciton_basis, test_3.n_exciton_basis)

    print(test_3.n_exciton_basis)
    print(_expected_four_exciton_basis)


def test_build_exciton_boson_basis():
    """
    Unit test for the build_exciton_boson_basis method.
    Tests against expected basis for three cases:

    (1) there is a single exciton and a boson with 2 levels
    (2) there is a single exciton and a boson with 3 levels
    (3) there are 2 excitons and a boson with 5 levels
    """
    sf = wptherml.SpectrumFactory()

    # dictionaries for 3 cases
    args_1 = {
        "number_of_excitons": 1,
        "number_of_boson_levels": 2,
    }

    args_2 = {
        "number_of_excitons": 1,
        "number_of_boson_levels": 3,
    }

    args_3 = {
        "number_of_excitons": 2,
        "number_of_boson_levels": 5,
    }

    # instantiate 3 cases
    test_1 = sf.spectrum_factory("Spin-Boson", args_1)
    test_2 = sf.spectrum_factory("Spin-Boson", args_2)
    test_3 = sf.spectrum_factory("Spin-Boson", args_3)

    # build the boson bases
    test_1.build_boson_basis()
    test_2.build_boson_basis()
    test_3.build_boson_basis()

    # build the exciton bases
    test_1.build_exciton_basis()
    test_2.build_exciton_basis()
    test_3.build_exciton_basis()

    # build the composite bases
    test_1.build_exciton_boson_basis()
    test_2.build_exciton_boson_basis()
    test_3.build_exciton_boson_basis()

    # create components of expected bases
    _expected_one_exciton_basis = np.matrix("1 0 ; 0 1")
    _expected_two_exciton_basis = np.kron(
        _expected_one_exciton_basis, _expected_one_exciton_basis
    )
    _expected_2_level_boson_basis = np.matrix("1 0 ; 0 1")
    _expected_3_level_boson_basis = np.matrix("1 0 0 ; 0 1 0 ; 0 0 1")
    _expected_5_level_boson_basis = np.matrix(
        "1 0 0 0 0 ; 0 1 0 0 0 ; 0 0 1 0 0 ; 0 0 0 1 0; 0 0 0 0 1"
    )

    # create expected composite basis for case 1
    _expected_exciton_boson_basis_case_1 = np.kron(
        _expected_2_level_boson_basis, _expected_one_exciton_basis
    )

    # create expected composite basis for case 2
    _expected_exciton_boson_basis_case_2 = np.kron(
        _expected_3_level_boson_basis, _expected_one_exciton_basis
    )

    # create expected composite basis for case 3
    _expected_exciton_boson_basis_case_3 = np.kron(
        _expected_5_level_boson_basis, _expected_two_exciton_basis
    )

    assert np.allclose(_expected_exciton_boson_basis_case_1, test_1.exciton_boson_basis)
    assert np.allclose(_expected_exciton_boson_basis_case_2, test_2.exciton_boson_basis)
    assert np.allclose(_expected_exciton_boson_basis_case_3, test_3.exciton_boson_basis)

    print(" Test 1 basis")
    print(test_1.exciton_boson_basis)
    print(" Expected test 1 basis")
    print(_expected_exciton_boson_basis_case_1)


def test_build_bosonic_ladder_operators():
    """
    Unit test for the build_bosonic_ladder_operator method

    (1) apply b to |1> when there is a 2-dimensional hilbert space and
        project onto <0|
    (2) apply b^+ to |0> when there is a 2-dimensional hilbert space and project
        onto <1|
    (3) apply b to |4> when there is a 6-dimensional hilbert space and
        project onto <3|
    (4) apply b^+ to |4> when there is a 6-dimensional hilbert space and
        project onto <5|
    (5) apply b^+ b to |8> when there is a 10 dimensional hilbert space and
        project onto <8|
    """
    # arguments for each case
    args_2d = {
        "number_of_boson_levels": 2,
    }
    args_6d = {
        "number_of_boson_levels": 6,
    }
    args_10d = {
        "number_of_boson_levels": 10,
    }

    sf = wptherml.SpectrumFactory()

    # instantiate cases
    test_2d = sf.spectrum_factory("Spin-Boson", args_2d)
    test_6d = sf.spectrum_factory("Spin-Boson", args_6d)
    test_10d = sf.spectrum_factory("Spin-Boson", args_10d)

    # build ladder oeprators
    test_2d.build_bosonic_ladder_operators()
    test_6d.build_bosonic_ladder_operators()
    test_10d.build_bosonic_ladder_operators()

    # create ket states for different cases
    _ket_1_2d = np.matrix("0 1").T
    _ket_0_2d = np.matrix("1 0").T

    _ket_4_6d = np.matrix("0 0 0 0 1 0").T

    _ket_8_10d = np.matrix("0 0 0 0 0 0 0 0 1 0").T

    _bra_0_2d = np.matrix("1 0")
    _bra_1_2d = np.matrix("0 1")

    _bra_3_6d = np.matrix("0 0 0 1 0 0")
    _bra_5_6d = np.matrix("0 0 0 0 0 1")

    _bra_8_10d = _ket_8_10d.T

    # compute <0|b|1> in 2D hilbert space
    _b_ket_1 = np.dot(test_2d.b_matrix, _ket_1_2d)
    _bra_0_2d_b_ket_1 = np.dot(_bra_0_2d, _b_ket_1)

    # compute <1|b^+|0> in 2D hilbert space
    _b_dag_ket_0 = np.dot(test_2d.b_dagger_matrix, _ket_0_2d)
    _bra_1_2d_b_ket_0 = np.dot(_bra_1_2d, _b_dag_ket_0)

    # both brackets should give 1
    assert np.isclose(_bra_0_2d_b_ket_1[0, 0], 1.0)
    assert np.isclose(_bra_1_2d_b_ket_0[0, 0], 1.0)

    # compute <3|b|4>
    _b_ket_4_6d = np.dot(test_6d.b_matrix, _ket_4_6d)
    _bra_3_6d_b_ket_4 = np.dot(_bra_3_6d, _b_ket_4_6d)

    # this bracket should give sqrt(4)
    assert np.isclose(_bra_3_6d_b_ket_4[0, 0], 2.0)

    # compute <5|b^+|4>
    _b_dag_ket_4_6d = np.dot(test_6d.b_dagger_matrix, _ket_4_6d)
    _bra_5_6d_b_dag_ket_4 = np.dot(_bra_5_6d, _b_dag_ket_4_6d)

    # this bracket should give sqrt(5)
    assert np.isclose(_bra_5_6d_b_dag_ket_4[0, 0], np.sqrt(5))

    # compute <8 | b^+ b | 8>
    b_dag_b_ket_8_10d = np.dot(
        test_10d.b_dagger_matrix, np.dot(test_10d.b_matrix, _ket_8_10d)
    )
    _bra_8_10d_b_dag_b_ket_8 = np.dot(_bra_8_10d, b_dag_b_ket_8_10d)

    # this bracket should be 8
    assert np.isclose(_bra_8_10d_b_dag_b_ket_8[0, 0], 8.0)


def test_compute_boson_energy_element():
    """
    Unit test for the compute_boson_energy_element method for a 1-exciton 2-level bosonic system with \hbar \omega = 6.8028 eV, compute the following elements of the boson Energy operator
        <0,0|H|0,0> = 1/2 \hbar \omega
            <1,0|H|1,0> = 3/2 \hbar \omega
            <0,1|H|0,1> = 1/2 \hbar \omega
            <1,1|H|1,1> = 3/2 \hbar \omega
            <1,0|H|0,1> = 0
    """

    # dictionaries for case 1
    args_1 = {
        "number_of_excitons": 1,
        "number_of_boson_levels": 2,
        "boson_energy_ev": 6.8028,
    }
    sf = wptherml.SpectrumFactory()

    # instantiate cases
    test_1 = sf.spectrum_factory("Spin-Boson", args_1)

    # build operators
    test_1.build_boson_basis()
    test_1.build_exciton_basis()
    test_1.build_exciton_boson_basis()
    test_1.build_bosonic_ladder_operators()
    test_1.build_boson_energy_operator()

    # create bra and kets
    _qd_bra_0 = np.matrix("1 0")
    _qd_bra_1 = np.matrix("0 1")
    _bos_bra_0 = np.matrix("1 0")
    _bos_bra_1 = np.matrix("0 1")

    _qd_ket_0 = _qd_bra_0.T
    _qd_ket_1 = _qd_bra_1.T

    _bos_ket_0 = _bos_bra_0.T
    _bos_ket_1 = _bos_bra_1.T

    _ket_00 = np.kron(_bos_ket_0, _qd_ket_0)
    _bra_00 = np.kron(_bos_bra_0, _qd_bra_0)

    _ket_10 = np.kron(_bos_ket_1, _qd_ket_0)
    _bra_10 = np.kron(_bos_bra_1, _qd_bra_0)

    _ket_01 = np.kron(_bos_ket_0, _qd_ket_1)
    _bra_01 = np.kron(_bos_bra_0, _qd_bra_1)

    _ket_11 = np.kron(_bos_ket_1, _qd_ket_1)
    _bra_11 = np.kron(_bos_bra_1, _qd_bra_1)

    _expected_E_0000 = test_1.boson_energy_au * 0.5
    _expected_E_1010 = test_1.boson_energy_au * 1.5
    _expected_E_1111 = test_1.boson_energy_au * 1.5
    _expected_E_0101 = test_1.boson_energy_au * 0.5
    _expected_E_1001 = 0.0

    _E_0000 = test_1.compute_boson_energy_element(_bra_00, _ket_00)
    _E_1010 = test_1.compute_boson_energy_element(_bra_10, _ket_10)
    _E_0101 = test_1.compute_boson_energy_element(_bra_01, _ket_01)
    _E_1111 = test_1.compute_boson_energy_element(_bra_11, _ket_11)

    _E_1001 = test_1.compute_boson_energy_element(_bra_10, _ket_01)

    assert np.isclose(_E_0000, _expected_E_0000)
    assert np.isclose(_E_1010, _expected_E_1010)
    assert np.isclose(_E_0101, _expected_E_0101)
    assert np.isclose(_E_1111, _expected_E_1111)
    assert np.isclose(_E_1001, _expected_E_1001)


def test_compute_boson_energy_matrix():
    """
    Unit test for the compute_boson_energy_element method for a 2-exciton 3-level bosonic system with \hbar \omega = 6.8028 eV.
    Compute matrix in the basis
        <q1, q2, s|H|q1',q2',s> =
    """

    # dictionaries for case 1
    args_1 = {
        "number_of_excitons": 2,
        "number_of_boson_levels": 3,
        "boson_energy_ev": 6.8028,
    }

    sf = wptherml.SpectrumFactory()

    # instantiate cases
    test_1 = sf.spectrum_factory("Spin-Boson", args_1)
    test_1.build_boson_basis()
    test_1.build_exciton_basis()
    test_1.build_exciton_boson_basis()
    test_1.build_bosonic_ladder_operators()
    test_1.build_boson_energy_operator()
    test_1.compute_boson_energy_matrix()

    _dim = 2**2 * 3
    _expected_matrix = np.zeros((_dim, _dim))

    # fill in boson energy elements using a loop over all basis states
    for i in range(3):
        for j in range(2):
            for k in range(2):
                _I = i * 2 * 2 + j * 2 + k
                for ip in range(3):
                    for jp in range(2):
                        for kp in range(2):
                            _J = ip * 2 * 2 + jp * 2 + kp
                            _expected_matrix[_I, _J] = (
                                (i + 1 / 2) * test_1.boson_energy_au
                            ) * (_I == _J)

    assert np.allclose(_expected_matrix, test_1.boson_energy_matrix)


def test_build_operator_for_exciton_j():
    """
        Unit test for the compute_operator_for_exciton_j method for the following test cases:

        (1a) There is 1 exciton system and we want to build the sigma^+ operator

        (1b) There is 1 exciton system and we want to build the sigma^- operator

        (1c) There is 1 exciton system and we want to build the sigma_z operator

        (2a) There are 2 exciton systems and we want to build sigma_z on the 1st one (index 0)

        (2b) There are 2 exciton systems and we want to build sigma_z on the 2nd one (index 1)

        (3a) There are 5 exciton systems and we want to build sigma_+ on the 3rd one (index 2)

        (3b) There are 5 exciton systems and we want to build sigma_- on the 2nd one (index 1)

    (3c) There are 5 exciton systems and we want to build sigma_x on the first one (index 0)

    (3d) There are 5 exciton systems and we want to build sigma_z on the last one (index 4)

    (3e) There are 5 exciton systems and we want to build sigma_+ sigma_- on the 4th one (index 3)

    """

    # dictionaries for case 1
    args_1 = {
        "number_of_excitons": 1,
    }
    # dictionaries for case 2
    args_2 = {
        "number_of_excitons": 2,
    }
    # dictionaries for case 3
    args_3 = {
        "number_of_excitons": 5,
    }

    # define some reference Pauli matrices
    _sigma_p = np.matrix("0 0 ; 1 0")  # sigma_p |0> == sigma_p [1 0].T = |1> == [0 1].T
    _sigma_m = np.matrix("0 1 ; 0 0")  # sigma_m |1> == sigma_m [0 1].T = |0> == [1 0].T
    _sigma_z = np.matrix("1 0 ; 0 -1")
    _sigma_x = np.matrix("0 1 ; 1 0")
    _ID = np.matrix("1 0 ; 0 1")
    _sigma_pm = np.matrix("0 0 ; 0 1")

    sf = wptherml.SpectrumFactory()

    # instantiate case 1
    test_1 = sf.spectrum_factory("Spin-Boson", args_1)
    test_1.build_exciton_basis()

    # create sigma^+ operator and test
    test_1.build_operator_for_exciton_j(0, "sigma_p")
    assert np.allclose(test_1.exciton_operator_j, _sigma_p)

    # create sigma^- operator and test
    test_1.build_operator_for_exciton_j(0, "sigma_m")
    assert np.allclose(test_1.exciton_operator_j, _sigma_m)

    # create sigma_z operator and test
    test_1.build_operator_for_exciton_j(0, "sigma_z")
    assert np.allclose(test_1.exciton_operator_j, _sigma_z)

    # instantiate case 2
    test_2 = sf.spectrum_factory("Spin-Boson", args_2)
    test_2.build_exciton_basis()

    # create sigma^+ operator on 1st and test
    test_2.build_operator_for_exciton_j(0, "sigma_z")
    _expected_op_1 = np.kron(_sigma_z, _ID)
    assert np.allclose(test_2.exciton_operator_j, _expected_op_1)

    # create sigma^+ operator on 2nd and test
    test_2.build_operator_for_exciton_j(1, "sigma_z")
    _expected_op_2 = np.kron(_ID, _sigma_z)
    assert np.allclose(test_2.exciton_operator_j, _expected_op_2)

    # instantiate case 3
    test_3 = sf.spectrum_factory("Spin-Boson", args_3)
    test_3.build_exciton_basis()

    # create sigma^+ operator on 3rd and test
    test_3.build_operator_for_exciton_j(2, "sigma_p")
    _expected_op_1 = np.kron(_ID, np.kron(_ID, np.kron(np.kron(_sigma_p, _ID), _ID)))
    assert np.allclose(test_3.exciton_operator_j, _expected_op_1)

    # create sigma^m operator on 2nd and test
    test_3.build_operator_for_exciton_j(1, "sigma_m")
    _expected_op_2 = np.kron(_ID, np.kron(_sigma_m, np.kron(np.kron(_ID, _ID), _ID)))
    assert np.allclose(test_3.exciton_operator_j, _expected_op_2)

    # create sigma_x operator on first and test
    test_3.build_operator_for_exciton_j(0, "sigma_x")
    _expected_op_3 = np.kron(_sigma_x, np.kron(_ID, np.kron(np.kron(_ID, _ID), _ID)))
    assert np.allclose(test_3.exciton_operator_j, _expected_op_3)

    # create sigma_z operator on last and test
    test_3.build_operator_for_exciton_j(4, "sigma_z")
    _expected_op_4 = np.kron(_ID, np.kron(_ID, np.kron(np.kron(_ID, _ID), _sigma_z)))
    assert np.allclose(test_3.exciton_operator_j, _expected_op_4)

    # create sigma_+ sigma_- on the 4th one and test
    test_3.build_operator_for_exciton_j(3, "sigma_pm")
    _expected_op_5 = np.kron(_ID, np.kron(_ID, np.kron(np.kron(_ID, _sigma_pm), _ID)))
    assert np.allclose(test_3.exciton_operator_j, _expected_op_5)


def test_compute_exciton_energy_element():
    # dictionaries for case 1
    args_1 = {
        "number_of_excitons": 2,
        "number_of_boson_levels": 2,
        "boson_energy_ev": 6.8028,
        "exciton_energy_ev": 6.8028 / 2.0,
    }
    sf = wptherml.SpectrumFactory()

    # instantiate cases
    test_1 = sf.spectrum_factory("Spin-Boson", args_1)
    test_1.exciton_energy_au = 0.5
    test_1.build_boson_basis()
    test_1.build_exciton_basis()
    test_1.build_exciton_boson_basis()
    test_1.build_exciton_energy_operator()

    _dim = test_1.exciton_boson_basis.shape[0]
    # create expected matrix
    _H = np.zeros((_dim, _dim))

    for i in range(_dim):
        _bra = test_1.exciton_boson_basis[:, i]
        for j in range(_dim):
            _ket = np.matrix(test_1.exciton_boson_basis[:, j]).T
            _element = test_1.compute_exciton_energy_element(_bra, _ket)
            _H[i, j] = _element

    # use method to compute matrix
    test_1.compute_exciton_energy_matrix()

    # compare elements
    assert np.allclose(_H, test_1.exciton_energy_matrix)
    

def test_compute_exciton_boson_coupling_matrix():
    """
    Unit test for the compute_exciton_boson_coupling_matrix method for a 2-exciton 3-level bosonic system with \hbar \g = 0.01 eV.
    Compute matrix in the basis
        <q1, q2, s|H|q1',q2',s> =
    """

    # dictionaries for case 1
    args_1 = {
        "number_of_excitons": 1,
        "number_of_boson_levels": 3,
        "boson_energy_ev": 6.8028,
        "exciton_boson_coupling_ev" : 0.01
    }

    sf = wptherml.SpectrumFactory()

    # instantiate cases
    test_1 = sf.spectrum_factory("Spin-Boson", args_1)
    test_1.build_boson_basis()
    test_1.build_exciton_basis()
    test_1.build_exciton_boson_basis()
    test_1.build_exciton_boson_coupling_operator()
    test_1.compute_exciton_boson_coupling_matrix()

    _dim = 6
    _expected_matrix = np.zeros((_dim, _dim))

    # fill in boson energy elements using a loop over all basis states
    for i in range(3): # bra boson index
        for j in range(2): # bra exciton index
            _I = i * 2 + j
            for ip in range(3): # ket boson index
                for jp in range(2): # ket exciton index 
                    _J = ip * 2 + jp 
                    if i==(ip+1) and j==(jp-1):
                         _expected_matrix[_I, _J] = test_1.exciton_boson_coupling_au * np.sqrt(ip+1)
                    elif i==(ip-1) and j==(jp+1):
                         _expected_matrix[_I, _J] = test_1.exciton_boson_coupling_au * np.sqrt(ip)

    assert np.allclose(_expected_matrix, test_1.exciton_boson_coupling_matrix)
    print("Expected Matrix")
    print(_expected_matrix)
    print("Built matrix")
    print(test_1.exciton_boson_coupling_matrix)


test_build_boson_basis()
test_build_exciton_basis()
test_build_exciton_boson_basis()
test_build_bosonic_ladder_operators()
test_compute_boson_energy_element()
test_compute_boson_energy_matrix()
test_build_operator_for_exciton_j()
test_compute_exciton_energy_element()
test_compute_exciton_boson_coupling_matrix()
