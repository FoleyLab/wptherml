"""
Unit and regression test for the wpspec package.
"""

# Import package, test suite, and other packages as needed
import wpspecdev
import numpy as np
import pytest
import sys

def test_wpspecdev_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "wpspecdev" in sys.modules

""" perform a series of tests that would apply to the simple
    case of a r = 100e-9 m spehere made of glass with ri = 1.5+0j
    in air with ri = 1.0+0j
"""
sf = wpspecdev.SpectrumFactory()
testargs = {
    'radius': 100e-9,
    'wavelength_list': [500e-9, 500e-9, 1]
}
# create test instance of mie
mietest = sf.spectrum_factory('Mie', testargs)
# create test instance of coefficient array
mietest._compute_n_array(mietest._size_factor_array[0])

def test_compute_mie_coefficients():
    #sf = wpspecdev.SpectrumFactory()
    #mt = sf.spectrum_factory('Mie', r)
    #mt._compute_mie_coeffients(ri, mu, x)

    expected_an = np.array([1.51561398e-07-3.89308843e-04j,
                            1.20867988e-13-3.47660737e-07j,
                            2.12941403e-20-1.45925119e-10j,
                            1.23296734e-27-3.51136346e-14j,
                            2.97642974e-35-5.45566654e-18j, 
                            3.49894198e-43-5.91518553e-22j,
                            2.23336769e-51-4.72585198e-26j])

    assert 1==1
    
def test_compute_s_jn():
    """ tests private method in MieDriver _compute_s_jn(n, z) """
    
    expected_result = np.array([3.56355664e-01, 9.39097524e-02, 1.72993686e-02, 2.45504861e-03,2.83621841e-04, 2.76413774e-05, 2.33017989e-06])
    
    result = mietest._compute_s_jn(mietest._n_array, mietest._size_factor_array[0])
    
    assert np.allclose(result, expected_result, 1e-5)
    

def test_compute_s_yn():
    """ test private method in MieDriver _compute_s_yn(n, z) """
    
    expected_result = np.array([-9.52514026e-01, -2.02805182e+00, -7.11684779e+00, -3.76158009e+01, -2.62286481e+02, -2.25831465e+03, -2.31001396e+04])
    
    result = mietest._compute_s_yn(mietest._n_array, mietest._size_factor_array[0])
    
    assert np.allclose(result, expected_result, 1e-5)
    
def test_compute_s_hn():
    """ test private method in MieDriver _compute_s_hn(n, z) """
    
    expected_result = np.array([3.56355664e-01-9.52514026e-01j, 9.39097524e-02-2.02805182e+00j,1.72993686e-02-7.11684779e+00j, 2.45504861e-03-3.76158009e+01j,2.83621841e-04-2.62286481e+02j, 2.76413774e-05-2.25831465e+03j, 2.33017989e-06-2.31001396e+04j])
    
    result = mietest._compute_s_hn(mietest._n_array, mietest._size_factor_array[0])
    
    assert np.allclose(result, expected_result, 1e-5)
    
def test_compute_z_hn_prime():
    """ test private method in MieDriver _compute_z_hn_prime(n, z) """
    
    expected_result = np.array([5.94700852e-01+6.43497032e-01j, 2.59990229e-01+2.85913922e+00j,6.61123694e-02+1.88020183e+01j, 1.19188333e-02+1.41519909e+02j, 1.66699586e-03+1.26416300e+03j, 1.90561452e-04+1.32202890e+04j,  1.84239201e-05+1.58863095e+05j])

    result = mietest._compute_z_hn_prime(mietest._n_array, mietest._size_factor_array[0])
    
    assert np.allclose(result, expected_result, 1e-5)