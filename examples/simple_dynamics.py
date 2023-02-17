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

""" Define a second test instance that will
    provide a simple case to test the dynamics... simple meaning
    no coupling, so the transition_dipole_moment is set to zero
    so that the V terms are 0 and so any localized exciton state
    (i.e. c_vector = [1, 0] or c_vector = [0, 1] will be an eigenstate)
    of the exciton Hamiltonian
"""
dynamics_args = {
'exciton_energy': 1.5,
'number_of_monomers' : 2,
'displacement_between_monomers' : np.array([1, 0, 0]),
'transition_dipole_moment' : np.array([0, 0, 0.0]),
'refractive_index' : 1.0

}

dynamics_test = sf.spectrum_factory("Frenkel", dynamics_args)
print(dynamics_test.exciton_hamiltonian)

#def test_rk_exciton():
#    ci = 0+1j
#
#    E1 = dynamics_test.exciton_energy

#    dt1 = 0.01

#    tf = 1

#    c_rk = np.array([1, 0], dtype=complex)

#    c_analytical = np.array([np.cos(E1) + ci * np.sin(E1), 0])

#    for i in range(1, 101):
#        dynamics_test._rk_exciton(dt1)

#    assert np.allclose(c_analytical, dynamics_test.c_vector)
