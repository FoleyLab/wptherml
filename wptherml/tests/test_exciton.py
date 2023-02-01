"""
Unit and regression test for the wpspec package.
"""

# Import package, test suite, and other packages as needed
import wptherml
import numpy as np
import pytest
import sys


""" perform a series of tests that would apply to the simple
    case of a r = 100e-9 m spehere made of glass with ri = 1.5+0j
    in air with ri = 1.0+0j
"""
sf = wptherml.SpectrumFactory()
args = {  
'exciton_energy': 1.5,
'number_of_monomers' : 10,
'displacement_between_monomers' : np.array([1, 0, 0]), 
'transition_dipole_moment' : np.array([0, 0, 0.5]) 
}  

exciton_test = sf.spectrum_factory('Frenkel', args)

def test_compute_H0_element():

    _H_11 = exciton_test._compute_H0_element(1, 1)
    _H_12 = exciton_test._compute_H0_element(1, 2)

    _expected_H_11 = exciton_test.exciton_energy
    _expected_H_12 = 0.

    assert np.isclose(_H_11, _expected_H_11, 1e-5)
    assert np.isclose(_H_12, _expected_H_12, 1e-5)


def test_compute_dipole_dipole_coupling():
    """add a real unit test here!"""

    pass


