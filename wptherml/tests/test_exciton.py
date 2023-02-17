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
'number_of_monomers' : 2,
'displacement_between_monomers' : np.array([1, 0, 0]),
'transition_dipole_moment' : np.array([0, 0, 0.5]),
'refractive_index' : 1.0

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

    _V_test = exciton_test._compute_dipole_dipole_coupling(1, 2)

    _V_expected = 0.25

    assert np.isclose(_V_test, _V_expected)



def test_build_exciton_hamiltonian():
    # test case based on 2x2 system defined in args above
    _H_expected = np.zeros((2,2))
    _H_expected[0,0] = exciton_test.exciton_energy
    _H_expected[1,1] = exciton_test.exciton_energy
    _H_expected[0,1] = 0.25
    _H_expected[1,0] = 0.25

    # this line will build the exciton hamiltonian and
    # store it in the attribute .exciton_hamiltonian
    exciton_test.build_exciton_hamiltonian()

    assert np.allclose(_H_expected, exciton_test.exciton_hamiltonian)

def test_rk_exciton():
    ci = 0+1j

    E1 = exciton_test.exciton_energy

    dt = 0.01

    tf = 1

    c_rk = np.array([1, 0], dtype=complex)

    n_basis = 2

    H_matrix = build_exciton_hamiltonian()

    c_analytical = np.array([np.exp(-ci * E1 * tf  ), 0+0j])

    for i in range(1, 101):
        c_rk4 = rk4_update(exciton_test.build_exciton_hamiltonian, c_rk4, dt)

    assert np.allclose(c_analytical, c_rk4)
