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

test_build_boson_basis()
test_build_exciton_basis()
test_build_exciton_boson_basis()
