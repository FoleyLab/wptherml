# validated against this google colab notebook with v1 of wptherml
# https://colab.research.google.com/drive/13YwoHzJwmNpElhZlYX8FphDImvaXDvQ4?usp=sharing 
import wpspecdev
from matplotlib import pyplot as plt
import numpy as np

test_args = {
    "wavelength_list": [300e-9, 6000e-9, 1000],
    "Material_List": ["Air", "Al2O3", "SiO2", "TiO2", "SiO2", "Al2O3", "W", "Air"],
    "Thickness_List": [0, 20e-9, 255e-9, 150e-9, 255e-9, 10e-9, 900e-9, 0],
    "temperature": 1700,
    "therml": True
}


def MaxwellGarnett(ri_1, ri_2, fraction):
    """ a function that will compute the alloy refractive
    index between material_1 and material_2 using
    Maxwell-Garnett theory.  """
    # define _eps_d as ri_1 ** 2
    _eps_d = ri_1 * ri_1

    # define _eps_m as ri_2 * ri_2
    _eps_m = ri_2 * ri_2

    # numerator of the Maxwell-Garnett model
    _numerator = _eps_d * (2 * fraction * (_eps_m - _eps_d) + _eps_m + 2 * _eps_d)
    # denominator of the Maxwell-Garnett model
    _denominator = 2 * _eps_d + _eps_m + fraction * (_eps_d - _eps_m)

    # _numerator / _denominator is epsilon_effective, and we want n_eff = sqrt(epsilon_eff)
    return np.sqrt(_numerator / _denominator)


sf = wpspecdev.SpectrumFactory()  
test = sf.spectrum_factory('Tmm', test_args)

print("original spectral efficiency")
print(test.stpv_spectral_efficiency)

# make this layer 70% W in alumina
n_eff = MaxwellGarnett(test._refractive_index_array[:,1], test._refractive_index_array[:,6], 0.75)
test._refractive_index_array[:,1] = n_eff

#test.compute_spectrum()
#test._compute_therml_spectrum(test.wavelength_array, test.emissivity_array)
#test._compute_stpv_power_density(test.wavelength_array)
#test._compute_stpv_spectral_efficiency(test.wavelength_array)

test.compute_stpv()
test.compute_stpv_gradient()
print(test.power_density_gradient)
print(test.stpv_spectral_efficiency_gradient)

#print(test.lambda_bandgap)
#print("Printing STPV Figures of Merit")
#print(test.stpv_power_density)
print("updated spectral efficiency")
print(test.stpv_spectral_efficiency)

plt.plot(test.wavelength_array, test.thermal_emission_array, 'red')
plt.plot(test.wavelength_array, test.blackbody_spectrum, 'black')
plt.show()


