"""
Unit and regression test for the wpspec package.
"""

# Import package, test suite, and other packages as needed
import wpspecdev
import numpy as np
import pytest
import sys


# define basic structure at 1500 K
test_args = {  
'wavelength_list': [100e-9, 30000e-9, 10000],  
'material_list': ["Air", "W", "Air"],
'thickness_list': [0, 800e-9, 0],
'temperature' : 1500, 
'therml' : True
}  

sf = wpspecdev.SpectrumFactory()  
test = sf.spectrum_factory('Tmm', test_args)



def test_compute_power_density():
    """ will test _compute_power_density method to 
        see if the power density computed by integration
        of the blackbody spectrum is close to the prediction
        by the Stefan-Boltzmann law, where the latter should be exact
        """
    assert np.isclose(test.blackbody_power_density, test.stefan_boltzmann_law, 1e-2)
