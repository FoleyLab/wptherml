"""
Unit and regression test for the wpspec package.
"""

# Import package, test suite, and other packages as needed
import wptherml
import numpy as np
import pytest
import sys
from numpy import linalg
import json
import os

path_and_file = os.path.realpath(__file__)
path = path_and_file[:-23] + "data/json_data/"


# high-level tests for spin-boson model and dynamics
def test_compute_spectrum():

    # test example vs QuTip J-C model for weak coupling between 1 spin and 2 level cavity
    # from json file
    test_1_json = path + "JC_simulation_RWA=_True-spin_freq_0.5_cavity_freq_0.5_cavity_coupling_0.0.json"
    test_2_json = path + "JC_simulation_RWA=_True-spin_freq_0.5_cavity_freq_0.5_cavity_coupling_0.02.json"

    # load json data
    with open(test_1_json) as f:
        json_1_data = json.load(f)

    with open(test_2_json) as f:
        json_2_data = json.load(f)

    

    # test example vs QuTip J-C model for strong coupling between 1 spin and 2 level cavity
    # with spin = cavity omega = 0.5 atomic units, g = 0.02 (using RWA)
    test_args_1 = {
     "Number_of_Excitons": 1,
     "number_of_boson_levels": json_1_data["number_of_cavity_states"],
     "boson_energy_ev": json_1_data["cavity_frequency"] / 3.6749322175665e-2, 
     "exciton_energy_ev" : json_1_data["spin_frequency"] / 3.6749322175665e-2, 
     "exciton_boson_coupling_ev" : json_1_data["cavity_coupling"] / 3.6749322175665e-2
     
    }

    test_args_2 = {
        "Number_of_Excitons": 1,
        "number_of_boson_levels": json_2_data["number_of_cavity_states"],
        "boson_energy_ev": json_2_data["cavity_frequency"] / 3.6749322175665e-2, 
        "exciton_energy_ev" : json_2_data["spin_frequency"] / 3.6749322175665e-2, 
        "exciton_boson_coupling_ev" : json_2_data["cavity_coupling"] / 3.6749322175665e-2
        
        }
    
    test_args_3 = {
     "Number_of_Excitons": 1,
     "number_of_boson_levels": 10,
     "boson_energy_ev": 0.5 / 3.6749322175665e-2, 
     "exciton_energy_ev" : 0.5 / 3.6749322175665e-2, 
     "exciton_boson_coupling_ev" : 0.49 / 3.6749322175665e-2
     
    }

    sf = wptherml.SpectrumFactory()
    
    # instantiate the three tests
    test_1 = sf.spectrum_factory("Spin-Boson", test_args_1)
    test_2 = sf.spectrum_factory("Spin-Boson", test_args_2)
    test_3 = sf.spectrum_factory("Spin-Boson", test_args_3)
    
    # get expected results for the three cases
    _expected_eigs_1 = np.array(json_1_data["energies"])
    _expected_eigs_2 = np.array(json_2_data["energies"])
    _expected_eigs_3 = np.array([0.0, 
                                 0.010000000000000009, 
                                 0.30703535443718355, 
                                 0.6512951042912501, 
                                 0.99, 
                                 1.02, 
                                 1.4043266910251035, 
                                 1.6929646455628165, 
                                 1.799750026036243, 
                                 2.2035818575783503, 
                                 2.34870489570875, 
                                 2.6140707088743675, 
                                 2.98, 
                                 3.0300000000000007, 
                                 3.595673308974897, 
                                 4.200249973963757, 
                                 4.79641814242165, 
                                 5.0, 
                                 5.385929291125633, 
                                 5.97])

    # wptherml includes zero-point energy but QuTip doesn't - we will subtract ZPE from ours
    test_1.energy_eigenvalues -= test_1.boson_energy_au / 2
    test_2.energy_eigenvalues -= test_2.boson_energy_au / 2
    test_3.energy_eigenvalues -= test_3.boson_energy_au / 2

    assert np.allclose(test_1.energy_eigenvalues, _expected_eigs_1)
    assert np.allclose(test_2.energy_eigenvalues, _expected_eigs_2)
    assert np.allclose(test_3.energy_eigenvalues, _expected_eigs_3)


def test_spin_boson_dynamics():
    test_args_1 = {
     "Number_of_Excitons": 1,
     "number_of_boson_levels": 2,
     "boson_energy_ev": 0.5 / 3.6749322175665e-2, 
     "exciton_energy_ev" : 0.5 / 3.6749322175665e-2, 
     "exciton_boson_coupling_ev" : 0.02 / 3.6749322175665e-2,
     "time_step_au" : 1.0
     
    }

    # test example vs QuTip J-C model for strong coupling between 1 spin and 2 level cavity
    # with spin = cavity omega = 0.5 atomic units, g = 0.02 (using RWA) and gamma = 5e-6 atomic units, kappa = 0
    # gamma is the spin decay rate in QuTip, kappa is the cavity decay rate
    # will set the initial condition corresponding to the spin in the excited state and cavity in the ground state

    # QuTip output for initial density matrix
    _expected_rho_init = np.array(
        [[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
         [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
         [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
         [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]]
    )

    # QuTip output for density matrix after 1000 time steps with dt = 1 atomic unit
    _expected_rho_t_10 = np.array(
        [[4.93374249e-05+0.j,        0.00000000e+00+0.j, 0.00000000e+00+0.j,        0.00000000e+00+0.j       ],
         [0.00000000e+00+0.j,        9.60482179e-01+0.j,0.00000000e+00+0.1947019j, 0.00000000e+00+0.j       ],
         [0.00000000e+00+0.j,        0.00000000e+00-0.1947019j, 3.94684840e-02+0.j,        0.00000000e+00+0.j       ],
         [0.00000000e+00+0.j,        0.00000000e+00+0.j, 0.00000000e+00+0.j,        0.00000000e+00+0.j       ]]

    )

    sf = wptherml.SpectrumFactory()
    


    # instantiate 
    test_1 = sf.spectrum_factory("Spin-Boson", test_args_1)

    

    # set the rates in atomic units by hand - can set it in eV before instantiation, but too much trouble to convert...
    test_1.exciton_spontaneous_emission_rate_au = 5e-6

    # set initial condition for wptherml
    initial_cav = np.array([[1],[0]])
    initial_sp = np.array([[0],[1]])
    initial_ket = np.kron(initial_cav, initial_sp)

    # assign initial rho to test_1.rho attribute
    test_1.rho = initial_ket @ initial_ket.conj().T

    # test against QuTip initial rho for the same state
    assert np.allclose(test_1.rho, _expected_rho_init)

    # propagate for 1000 timesteps
    for i in range(10):
        test_1.rk4_update_on_rho()


    # test against QuTip rho at t = 10
    assert np.allclose(test_1.rho, _expected_rho_t_10)


# unit tests
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
    _expected_basis = np.eye(3) #np.matrix("1 0 0 ; 0 1 0 ; 0 0 1")
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

    _expected_one_exciton_basis = np.eye(2)
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
    _expected_one_exciton_basis = np.eye(2)
    _expected_two_exciton_basis = np.kron(
        _expected_one_exciton_basis, _expected_one_exciton_basis
    )
    _expected_2_level_boson_basis = np.eye(2)
    _expected_3_level_boson_basis = np.eye(3)
    _expected_5_level_boson_basis = np.eye(5)

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
    _ket_1_2d = np.array([[0, 1]]).T #np.matrix("0 1").T
    _ket_0_2d = np.array([[1, 0]]).T #np.matrix("1 0").T

    _ket_4_6d = np.array([[0, 0, 0, 0, 1, 0]]).T #np.matrix("0 0 0 0 1 0").T

    _ket_8_10d = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]).T #np.matrix("0 0 0 0 0 0 0 0 1 0").T

    _bra_0_2d = np.array([[1, 0]])
    _bra_1_2d = np.array([[0, 1]])

    _bra_3_6d = np.array([[0, 0, 0, 1, 0, 0]])   #np.matrix("0 0 0 1 0 0")
    _bra_5_6d = np.array([[0, 0, 0, 0, 0, 1]])   #np.matrix("0 0 0 0 0 1")

    _bra_8_10d = _ket_8_10d.T

    # compute <0|b|1> in 2D hilbert space
    _b_ket_1 = np.dot(test_2d.b_matrix, _ket_1_2d)
    _bra_0_2d_b_ket_1 = np.dot(_bra_0_2d, _b_ket_1)

    # compute <1|b^+|0> in 2D hilbert space
    _b_dag_ket_0 = np.dot(test_2d.b_dagger_matrix, _ket_0_2d)
    _bra_1_2d_b_ket_0 = np.dot(_bra_1_2d, _b_dag_ket_0)

    # both brackets should give 1
    assert np.isclose(_bra_0_2d_b_ket_1[0], 1.0)
    assert np.isclose(_bra_1_2d_b_ket_0[0], 1.0)

    # compute <3|b|4>
    _b_ket_4_6d = np.dot(test_6d.b_matrix, _ket_4_6d)
    _bra_3_6d_b_ket_4 = np.dot(_bra_3_6d, _b_ket_4_6d)

    # this bracket should give sqrt(4)
    assert np.isclose(_bra_3_6d_b_ket_4[0], 2.0)

    # compute <5|b^+|4>
    _b_dag_ket_4_6d = np.dot(test_6d.b_dagger_matrix, _ket_4_6d)
    _bra_5_6d_b_dag_ket_4 = np.dot(_bra_5_6d, _b_dag_ket_4_6d)

    # this bracket should give sqrt(5)
    assert np.isclose(_bra_5_6d_b_dag_ket_4[0], np.sqrt(5))

    # compute <8 | b^+ b | 8>
    b_dag_b_ket_8_10d = np.dot(
        test_10d.b_dagger_matrix, np.dot(test_10d.b_matrix, _ket_8_10d)
    )
    _bra_8_10d_b_dag_b_ket_8 = np.dot(_bra_8_10d, b_dag_b_ket_8_10d)

    # this bracket should be 8
    assert np.isclose(_bra_8_10d_b_dag_b_ket_8[0], 8.0)



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
    _sigma_p = np.array([[0, 0], [1, 0]])   #np.matrix("0 0 ; 1 0")  # sigma_p |0> == sigma_p [1 0].T = |1> == [0 1].T
    _sigma_m = np.array([[0, 1], [0, 0]])   #np.matrix("0 1 ; 0 0")  # sigma_m |1> == sigma_m [0 1].T = |0> == [1 0].T
    _sigma_z = np.array([[1, 0], [0, -1]])  #np.matrix("1 0 ; 0 -1")
    _sigma_x = np.array([[0, 1], [1, 0]])   #np.matrix("0 1 ; 1 0")
    _ID = np.eye(2) # np.matrix("1 0 ; 0 1")
    _sigma_pm = np.array([[0, 0], [0, 1]])  #np.matrix("0 0 ; 0 1")

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

    assert np.allclose(_expected_matrix, test_1.exciton_boson_coupling_operator)
    print("Expected Matrix")
    print(_expected_matrix)
    print("Built matrix")
    print(test_1.exciton_boson_coupling_operator)


test_build_boson_basis()
test_build_exciton_basis()
test_build_exciton_boson_basis()
test_build_bosonic_ladder_operators()
#test_compute_boson_energy_matrix()
test_build_operator_for_exciton_j()
test_compute_exciton_boson_coupling_matrix()
