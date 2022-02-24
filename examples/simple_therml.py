import wpspecdev
from matplotlib import pyplot as plt
import numpy as np

test_args = {
    "wavelength_list": [300e-9, 60000e-9, 5000],
    "material_list": ["Air", "SiO2", "Air"],
    "thickness_list": [0, 230e-9, 0],
    "temperature": 300,
    "cooling": True,
}


#args = {  
#'wavelength_list': [500e-9, 502e-9, 3],  
#'material_list': ["Air", "Ag", "Air"],
#'thickness_list': [0,  230e-9, 0],
#'TeMperature' : 5000, 
#'therml': True
#}  

sf = wpspecdev.SpectrumFactory()  
test = sf.spectrum_factory('Tmm', test_args)
print("atmospheric cooling", test.atmospheric_radiated_power)
test._refractive_index_array[:,1] = 2.4+0.2j

# get \epsilon_s(\lambda, \theta) and \epsilon_s(\lambda, \theta) for thermal radiation
test.compute_explicit_angle_spectrum()

# call _compute_thermal_radiated_power( ) function
test.thermal_radiated_power = test._compute_thermal_radiated_power(test.emissivity_array_s,test.emissivity_array_p, test.theta_vals, test.theta_weights, test.wavelength_array)
print(test.thermal_radiated_power)


# get \epsilon_s(\lambda, \theta) and \epsilon_s(\lambda, \theta) for thermal radiation
test.compute_explicit_angle_spectrum()

# call _compute_thermal_radiated_power( ) function 
test.atmospheric_radiated_power = test._compute_atmospheric_radiated_power(test._atmospheric_transmissivity, test.emissivity_array_s, test.emissivity_array_p, test.theta_vals, test.theta_weights, test.wavelength_array)
print("Just updated this thing!")
print(test.atmospheric_radiated_power)
"""
test._refractive_index_array[0,1] = 0.04998114+3.13267845j
test._refractive_index_array[1,1] = 0.04997774+3.14242568j
test._refractive_index_array[2,1] = 0.04997488+3.15210009j
test.compute_explicit_angle_spectrum()
print(test.reflectivity_array_p[:,1])
"""
#print(test.reflectivity_array_s)
#print(test.blackbody_power_density)
#print(test.stefan_boltzmann_law)
#print(test.stpv_power_density)

#plt.plot(test.wavelength_array, test._solar_spectrum/1.4e9)
#plt.plot(test.wavelength_array, test._atmospheric_transmissivity)
#plt.show()




