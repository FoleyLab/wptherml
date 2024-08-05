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
	sf = wptherml.SpectrumFactory()
	args = {
		'number_of_boson_levels' : 3,
	}
	
	test = sf.spectrum_factory("Spin-Boson", args)
	test.build_boson_basis()
	_expected_basis = np.matrix('1 0 0 ; 0 1 0 ; 0 0 1')
	assert np.allclose(_expected_basis, test.boson_basis)
	print(test.boson_basis)


test_build_boson_basis()
