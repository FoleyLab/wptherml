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
		'number_of_boson_levels' : 3,
	}
	
	test = sf.spectrum_factory("Spin-Boson", args)
	test.build_boson_basis()
	_expected_basis = np.matrix('1 0 0 ; 0 1 0 ; 0 0 1')
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
		'number_of_excitons' : 1,
	}

	args_2 = {
		'number_of_excitons' : 2,
	}

	args_3 = {
		'number_of_excitons' : 4,
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
	_expected_two_exciton_basis = np.kron(_expected_one_exciton_basis, _expected_one_exciton_basis)
	_expected_four_exciton_basis = np.kron(_expected_one_exciton_basis, np.kron(_expected_one_exciton_basis, _expected_two_exciton_basis))

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
		'number_of_excitons' : 1,
		'number_of_boson_levels' : 2,

	}

	args_2 = {
		'number_of_excitons' : 1,
		'number_of_boson_levels' : 3,
	}

	args_3 = {
		'number_of_excitons' : 2,
		'number_of_boson_levels' : 5,
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
	_expected_two_exciton_basis = np.kron(_expected_one_exciton_basis, _expected_one_exciton_basis)
	_expected_2_level_boson_basis = np.matrix("1 0 ; 0 1")
	_expected_3_level_boson_basis = np.matrix("1 0 0 ; 0 1 0 ; 0 0 1")
	_expected_5_level_boson_basis = np.matrix("1 0 0 0 0 ; 0 1 0 0 0 ; 0 0 1 0 0 ; 0 0 0 1 0; 0 0 0 0 1")

	# create expected composite basis for case 1
	_expected_exciton_boson_basis_case_1 = np.kron(_expected_2_level_boson_basis, _expected_one_exciton_basis)

	# create expected composite basis for case 2
	_expected_exciton_boson_basis_case_2 = np.kron(_expected_3_level_boson_basis, _expected_one_exciton_basis)

	# create expected composite basis for case 3
	_expected_exciton_boson_basis_case_3 = np.kron(_expected_5_level_boson_basis, _expected_two_exciton_basis)

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
		'number_of_boson_levels' : 2,
	}
	args_6d = {
		'number_of_boson_levels' : 6,
	}
	args_10d = {
		'number_of_boson_levels' : 10,
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
	_ket_1_2d = np.matrix('0 1').T
	_ket_0_2d = np.matrix('1 0').T

	_ket_4_6d = np.matrix('0 0 0 0 1 0').T

	_ket_8_10d = np.matrix('0 0 0 0 0 0 0 0 1 0').T

	_bra_0_2d = np.matrix('1 0')
	_bra_1_2d = np.matrix('0 1')

	_bra_3_6d = np.matrix('0 0 0 1 0 0')
	_bra_5_6d = np.matrix('0 0 0 0 0 1')

	_bra_8_10d = _ket_8_10d.T

	# compute <0|b|1> in 2D hilbert space
	_b_ket_1 = np.dot(test_2d.b_matrix, _ket_1_2d) 
	_bra_0_2d_b_ket_1 = np.dot(_bra_0_2d, _b_ket_1)

	# compute <1|b^+|0> in 2D hilbert space
	_b_dag_ket_0 = np.dot(test_2d.b_dagger_matrix, _ket_0_2d) 
	_bra_1_2d_b_ket_0 = np.dot(_bra_1_2d, _b_dag_ket_0)

	# both brackets should give 1
	assert np.isclose(_bra_0_2d_b_ket_1[0,0], 1.0)
	assert np.isclose(_bra_1_2d_b_ket_0[0,0], 1.0)

	# compute <3|b|4>
	_b_ket_4_6d = np.dot(test_6d.b_matrix, _ket_4_6d)
	_bra_3_6d_b_ket_4 = np.dot(_bra_3_6d, _b_ket_4_6d)

	# this bracket should give sqrt(4)
	assert np.isclose(_bra_3_6d_b_ket_4[0,0], 2.0)

	# compute <5|b^+|4>
	_b_dag_ket_4_6d = np.dot(test_6d.b_dagger_matrix, _ket_4_6d)
	_bra_5_6d_b_dag_ket_4 = np.dot(_bra_5_6d, _b_dag_ket_4_6d)

	# this bracket should give sqrt(5)
	assert np.isclose(_bra_5_6d_b_dag_ket_4[0,0], np.sqrt(5))

	# compute <8 | b^+ b | 8>
	b_dag_b_ket_8_10d = np.dot(test_10d.b_dagger_matrix, np.dot(test_10d.b_matrix, _ket_8_10d))
	_bra_8_10d_b_dag_b_ket_8 = np.dot(_bra_8_10d, b_dag_b_ket_8_10d)

	# this bracket should be 8
	assert np.isclose(_bra_8_10d_b_dag_b_ket_8[0,0], 8.0)


def test_compute_boson_energy_element():
	"""
	Unit test for the compute_boson_energy_element method

	(1) For a 1-exciton 2-level bosonic system with \hbar \omega = 6.8028 eV, compute the following elements of the boson Energy operator
	    <0,0|H|0,0> = 1/2 
	"""
	
	# dictionaries for case 1
	args_1 = {
		'number_of_excitons' : 1,
		'number_of_boson_levels' : 2,
		'boson_energy_ev' : 6.8028,

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
	_qd_bra_0 = np.matrix('1 0')
	_qd_bra_1 = np.matrix('0 1')
	_bos_bra_0 = np.matrix('1 0')
	_bos_bra_1 = np.matrix('0 1')

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


	_E_0000 = test_1.compute_boson_energy_element(_bra_00, _ket_00)
	_E_1010 = test_1.compute_boson_energy_element(_bra_10, _ket_10)
	_E_0101 = test_1.compute_boson_energy_element(_bra_01, _ket_01)
	_E_1111 = test_1.compute_boson_energy_element(_bra_11, _ket_11)

	print(F' <00|H|00> : {_E_0000}')
	print(F' <10|H|10> : {_E_1010}')
	print(F' <01|H|01> : {_E_0101}')
	print(F' <11|H|11> : {_E_1111}')

	assert np.isclose(_E_0000, _expected_E_0000)
	assert np.isclose(_E_1010, _expected_E_1010)
	assert np.isclose(_E_0101, _expected_E_0101)
	assert np.isclose(_E_1111, _expected_E_1111)



	


	

test_build_boson_basis()
test_build_exciton_basis()
test_build_exciton_boson_basis()
test_build_bosonic_ladder_operators()
test_compute_boson_energy_element()