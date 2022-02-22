import wpspecdev
from matplotlib import pyplot as plt
import numpy as np
args = {  
'wavelength_list': [500e-9, 502e-9, 3],  
'material_list': ["Air", "Ag", "Air"],
'thickness_list': [0,  230e-9, 0],
'TeMperature' : 5000, 
'therml': True
}  

sf = wpspecdev.SpectrumFactory()  
test = sf.spectrum_factory('Tmm', args)

test._refractive_index_array[0,1] = 0.04998114+3.13267845j
test._refractive_index_array[1,1] = 0.04997774+3.14242568j
test._refractive_index_array[2,1] = 0.04997488+3.15210009j
test.compute_explicit_angle_spectrum()
print(test.reflectivity_array_p[:,1])
#print(test.reflectivity_array_s)
#print(test.blackbody_power_density)
#print(test.stefan_boltzmann_law)
#print(test.stpv_power_density)

#plt.plot(test.wavelength_array, test._solar_spectrum/1.4e9)
#plt.plot(test.wavelength_array, test._atmospheric_transmissivity)
#plt.show()




