"""
Unit and regression test for the wpspec package.
"""

# Import package, test suite, and other packages as needed
import wpspecdev
import numpy as np
import pytest
import sys


""" perform a series of tests that would apply to the simple
    case of a r = 100e-9 m spehere made of glass with ri = 1.5+0j
    in air with ri = 1.0+0j
"""
sf = wpspecdev.SpectrumFactory()
testargs = {"wavelength_list": [400e-9, 800e-9, 10]}
# create test instance of mie
mietest = sf.spectrum_factory("Tmm", testargs)


def test_compute_spectrum():
    """tests public method in TmmDriver compute_spectrum()
       current test system is Air / 900 nm SiO2 / Air 
       where SiO2 is modelled as static refractive index of n = 1.5 + 0j.
       Should elaborate more tests in the future!
    
    """
    expected_result = np.array(
        [
            0.0798722,
            0.00937256,
            0.14728703,
            0.0065643,  
            0.11280357,
            0.11737882,
            0.00423059,
            0.05803204,
            0.14127024,
            0.12906124
        ]
    )
    mietest.compute_spectrum()
    assert np.allclose(mietest.reflectivity_array, expected_result, 1e-5)