"""
Unit and regression test for the wpspec package.
"""

# Import package, test suite, and other packages as needed
import wptherml
import numpy as np
import pytest
import sys
from numpy import linalg


sf = wptherml.SpectrumFactory()
x_args = {
'exciton_energy': 1.5,
'aggregate_shape': (2,1,1),
'displacement_vector' : [1, 0, 0],  
'transition_dipole_moment' : np.array([0, 0, 0.5]),
'refractive_index' : 1.0,
}

exciton_x_test = sf.spectrum_factory('Frenkel', x_args)

y_args = {
'exciton_energy': 1.5,
'aggregate_shape': (1,2,1),
'displacement_vector' : [0, 1, 0],  
'transition_dipole_moment' : np.array([0, 0, 0.5]),
'refractive_index' : 1.0,
}

exciton_y_test = sf.spectrum_factory('Frenkel', y_args)

z_args = {
'exciton_energy': 1.5,
'aggregate_shape': (1,1,2),
'displacement_vector' : [0, 0, 1],  
'transition_dipole_moment' : np.array([0, 0, 0.5]),
'refractive_index' : 1.0,
}

exciton_z_test = sf.spectrum_factory('Frenkel', z_args)

xyz_args = {
'exciton_energy': 1.5,
'aggregate_shape': (1,1,2),
'displacement_vector' : [1, 1, 1],  
'transition_dipole_moment' : np.array([0, 0, 0.5]),
'refractive_index' : 1.0,
}

exciton_z_test = sf.spectrum_factory('Frenkel', xyz_args)

three_monomers_args = {
'exciton_energy': 1.5,
'aggregate_shape': (3,1,1),
'displacement_vector' : [1, 0, 0],  
'transition_dipole_moment' : np.array([0, 0, 0.5]),
'refractive_index' : 1.0,
}

exciton_three_test = sf.spectrum_factory('Frenkel', three_monomers_args)

four_monomers_args = {
'exciton_energy': 1.5,
'aggregate_shape': (2,2,1),
'displacement_vector' : [1, 1, 0],  
'transition_dipole_moment' : np.array([0, 0, 0.5]),
'refractive_index' : 1.0,
}

exciton_four_test = sf.spectrum_factory('Frenkel', four_monomers_args)


dynamics_args = {
'exciton_energy': 1,
'aggregate_shape': (2,1,1),
'displacement_vector' : [1000000, 0, 0],  
'transition_dipole_moment' : np.array([0, 0, 0.0]),
'refractive_index' : 1.0,
}

dynamics_test = sf.spectrum_factory("Frenkel", dynamics_args)

def test_compute_H0_element():

    _H_11 = exciton_x_test._compute_H0_element(1, 1)
    _H_12 = exciton_x_test._compute_H0_element(1, 2)

    _expected_H_11 = exciton_x_test.exciton_energy
    _expected_H_12 = 0.

    assert np.isclose(_H_11, _expected_H_11, 1e-5)
    assert np.isclose(_H_12, _expected_H_12, 1e-5)


def test_compute_dipole_dipole_coupling():

    _V_x_test = exciton_x_test._compute_dipole_dipole_coupling(0, 1)

    _V_x_expected = (
                1 / (exciton_x_test.refractive_index**2 * np.sqrt(np.dot(exciton_x_test.displacement_vector, exciton_x_test.displacement_vector)) ** 3)
            ) * (
                np.dot(exciton_x_test.transition_dipole_moment, exciton_x_test.transition_dipole_moment)
                - 3
                * (
                    (
                        np.dot(exciton_x_test.transition_dipole_moment, exciton_x_test.displacement_vector)
                        * np.dot(exciton_x_test.displacement_vector, exciton_x_test.transition_dipole_moment)
                    )
                    / (np.sqrt(np.dot(exciton_x_test.displacement_vector, exciton_x_test.displacement_vector)) ** 2)
                )
            )
    
    _V_y_test = exciton_x_test._compute_dipole_dipole_coupling(0, 1)

    _V_y_expected = (
                1 / (exciton_x_test.refractive_index**2 * np.sqrt(np.dot(exciton_x_test.displacement_vector, exciton_x_test.displacement_vector)) ** 3)
            ) * (
                np.dot(exciton_x_test.transition_dipole_moment, exciton_x_test.transition_dipole_moment)
                - 3
                * (
                    (
                        np.dot(exciton_x_test.transition_dipole_moment, exciton_x_test.displacement_vector)
                        * np.dot(exciton_x_test.displacement_vector, exciton_x_test.transition_dipole_moment)
                    )
                    / (np.sqrt(np.dot(exciton_x_test.displacement_vector, exciton_x_test.displacement_vector)) ** 2)
                )
            )
    
    _V_z_test = exciton_x_test._compute_dipole_dipole_coupling(0, 1)

    _V_z_expected = (
                1 / (exciton_x_test.refractive_index**2 * np.sqrt(np.dot(exciton_x_test.displacement_vector, exciton_x_test.displacement_vector)) ** 3)
            ) * (
                np.dot(exciton_x_test.transition_dipole_moment, exciton_x_test.transition_dipole_moment)
                - 3
                * (
                    (
                        np.dot(exciton_x_test.transition_dipole_moment, exciton_x_test.displacement_vector)
                        * np.dot(exciton_x_test.displacement_vector, exciton_x_test.transition_dipole_moment)
                    )
                    / (np.sqrt(np.dot(exciton_x_test.displacement_vector, exciton_x_test.displacement_vector)) ** 2)
                )
            )
    
    _V_xyz_test = exciton_x_test._compute_dipole_dipole_coupling(0, 1)

    _V_xyz_expected = (
                1 / (exciton_x_test.refractive_index**2 * np.sqrt(np.dot(exciton_x_test.displacement_vector, exciton_x_test.displacement_vector)) ** 3)
            ) * (
                np.dot(exciton_x_test.transition_dipole_moment, exciton_x_test.transition_dipole_moment)
                - 3
                * (
                    (
                        np.dot(exciton_x_test.transition_dipole_moment, exciton_x_test.displacement_vector)
                        * np.dot(exciton_x_test.displacement_vector, exciton_x_test.transition_dipole_moment)
                    )
                    / (np.sqrt(np.dot(exciton_x_test.displacement_vector, exciton_x_test.displacement_vector)) ** 2)
                )
            )

    assert np.isclose(_V_x_test, _V_x_expected)
    assert np.isclose(_V_y_test, _V_y_expected)
    assert np.isclose(_V_z_test, _V_z_expected)
    assert np.isclose(_V_xyz_test, _V_xyz_expected)



def test_build_exciton_hamiltonian():
    # test case based on 2x2 system defined in args above
    _H_three_expected = np.zeros((3,3))
    _H_three_expected[0,0] = exciton_three_test.exciton_energy
    _H_three_expected[1,1] = exciton_three_test.exciton_energy
    _H_three_expected[2,2] = exciton_three_test.exciton_energy
    _H_three_expected[0,1] = exciton_three_test._compute_dipole_dipole_coupling(0, 1)
    _H_three_expected[1,0] = exciton_three_test._compute_dipole_dipole_coupling(1, 0)
    _H_three_expected[2,0] = exciton_three_test._compute_dipole_dipole_coupling(2, 0)
    _H_three_expected[2,1] = exciton_three_test._compute_dipole_dipole_coupling(2, 1)
    _H_three_expected[1,2] = exciton_three_test._compute_dipole_dipole_coupling(1, 2)
    _H_three_expected[0,2] = exciton_three_test._compute_dipole_dipole_coupling(0, 2)

    # this line will build the exciton hamiltonian and
    # store it in the attribute .exciton_hamiltonian
    exciton_three_test.build_exciton_hamiltonian()

    _H_four_expected = np.zeros((4,4))
    _H_four_expected[0,0] = exciton_four_test.exciton_energy
    _H_four_expected[1,1] = exciton_four_test.exciton_energy
    _H_four_expected[2,2] = exciton_four_test.exciton_energy
    _H_four_expected[3,3] = exciton_four_test.exciton_energy
    _H_four_expected[0,1] = exciton_four_test._compute_dipole_dipole_coupling(0, 1)
    _H_four_expected[1,0] = exciton_four_test._compute_dipole_dipole_coupling(1, 0)
    _H_four_expected[2,0] = exciton_four_test._compute_dipole_dipole_coupling(2, 0)
    _H_four_expected[2,1] = exciton_four_test._compute_dipole_dipole_coupling(2, 1)
    _H_four_expected[1,2] = exciton_four_test._compute_dipole_dipole_coupling(1, 2)
    _H_four_expected[0,2] = exciton_four_test._compute_dipole_dipole_coupling(0, 2)
    _H_four_expected[0,3] = exciton_four_test._compute_dipole_dipole_coupling(0, 3)
    _H_four_expected[1,3] = exciton_four_test._compute_dipole_dipole_coupling(1, 3)
    _H_four_expected[2,3] = exciton_four_test._compute_dipole_dipole_coupling(2, 3)
    _H_four_expected[3,2] = exciton_four_test._compute_dipole_dipole_coupling(3, 2)
    _H_four_expected[3,1] = exciton_four_test._compute_dipole_dipole_coupling(3, 1)
    _H_four_expected[3,0] = exciton_four_test._compute_dipole_dipole_coupling(3, 0)

    exciton_three_test.build_exciton_hamiltonian()
    exciton_four_test.build_exciton_hamiltonian()

    assert np.allclose(_H_three_expected, exciton_three_test.exciton_hamiltonian)
    assert np.allclose(_H_four_expected, exciton_four_test.exciton_hamiltonian)


def test_rk_exciton():
    ci = 0+1j

    E1 = dynamics_test.exciton_energy
    dynamics_test.build_exciton_hamiltonian()

    dt1 = 0.01

    tf = 1

    c_analytical = np.array([[np.cos(E1) - ci * np.sin(E1)], [0 + 0j]])

    for i in range(1, 101):
        dynamics_test._rk_exciton(dt1)

    #assert np.allclose(c_analytical, dynamics_test.c_vector)
    pass 

def test_rk_density_matrix():
    ci = 0+1j

    E1 = dynamics_test.exciton_energy
    E2 = dynamics_test.exciton_energy
    dynamics_test.build_exciton_hamiltonian()
    #test_H = dynamics_test.exciton_hamiltonian
    #test_diag = np.linalg.eigh(test_H)
    #V_1 = test_diag.eigenvectors[:,0]
    #V_2 = test_diag.eigenvectors[:,1]




    #dt1 = 0.01

    #tf = 1

    #D_analytical = np.array([[1 * V_1, 0 * V_2], [1 * V_1, 0 * V_2]])

    #for i in range(1, 101):
    #    dynamics_test._rk_exciton_density_matrix(dt1)

    #assert np.allclose(D_analytical, dynamics_test.density_matrix)
    pass

