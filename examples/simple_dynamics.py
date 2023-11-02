# Import package, test suite, and other packages as needed
import wptherml
import numpy as np
import pytest
import sys


sf = wptherml.SpectrumFactory()
args = {
'exciton_energy': 1.5,
'number_of_monomers' : 2,
'displacement_between_monomers' : np.array([1, 0, 0]),
'transition_dipole_moment' : np.array([0, 0, 0.5]),
'refractive_index' : 1.0

}

exciton_test = sf.spectrum_factory('Frenkel', args)

dynamics_args = {
'exciton_energy': 1.5,
'number_of_monomers' : 2,
'displacement_between_monomers' : np.array([1, 0, 0]),
'transition_dipole_moment' : np.array([0, 0, 0.0]),
'refractive_index' : 1.0

}

dynamics_test = sf.spectrum_factory("Frenkel", dynamics_args)
print(dynamics_test.exciton_hamiltonian)


