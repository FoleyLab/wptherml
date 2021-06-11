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

    expected_an = np.array([1.09393760e-01-3.12132609e-01j, 9.58682444e-04-3.09477523e-02j,
 1.63054752e-06-1.27692790e-03j, 9.66113753e-10-3.10823704e-05j,
 2.39854703e-13-4.89749633e-07j, 2.88856083e-17-5.37453331e-09j,
 1.88087968e-21-4.33691097e-11j])

    expected_bn = np.array([8.51059716e-03-9.18594954e-02j, 1.35466949e-05-3.68055857e-03j,
 8.15604142e-09-9.03108042e-05j, 2.07821108e-12-1.44160018e-06j,
 2.56048144e-16-1.60015044e-08j, 1.69880890e-20-1.30338363e-10j,
 6.58730141e-25-8.11621920e-13j] )
    
    expected_cn = np.array([1.02133636+9.46247560e-02j, 0.5558344 +2.04580879e-03j,
 0.34394533+3.10619798e-05j, 0.22121405+3.18902215e-07j,
 0.14432436+2.30940695e-09j, 0.09482464+1.23592890e-11j,
 0.0625545 +5.07706007e-14j])
    expected_dn = np.array([0.94943955+3.32752039e-01j, 0.57237172+1.77306162e-02j,
 0.33722182+4.30608658e-04j, 0.21300494+6.62069833e-06j,
 0.13775492+6.74654196e-08j, 0.09001634+4.83795816e-10j,
 0.05915672+2.56557430e-12j] )
    check_lst = [expected_an, expected_bn, expected_cn, expected_dn]
    #for i in check_lst:
    result = mietest._compute_mie_coeffients(mietest._relative_refractive_index_array, mietest._relative_permeability, mietest._size_factor_array)
    print(result[0],'\n'*3)
    assert np.allclose(result[0], expected_an, 1e-1)
        
    
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
    



def test_compute_z_jn_prime():
    """ test private method in MieDriver _compute_z_jn_prime(n,z)"""

    expected_result = np.array([5.94700852e-01, 2.59990229e-01, 6.61123694e-02, 1.19188333e-02,
 1.66699586e-03, 1.90561452e-04, 1.84239201e-05])

    result = mietest._compute_z_jn_prime(mietest._n_array, mietest._size_factor_array[0])

    assert np.allclose(result, expected_result, 1e-5)