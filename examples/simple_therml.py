import wpspecdev
from matplotlib import pyplot as plt
import numpy as np
args = {  
'wavelength_list': [400e-9, 20000e-9, 4000],  
'material_list': ["Air", "TiN", "Air"],
'thickness_list': [0,  400e-9, 0],
'TeMperature' : 5000, 
'therml': True
}  

sf = wpspecdev.SpectrumFactory()  
test = sf.spectrum_factory('Tmm', args)

print(test.blackbody_power_density)
print(test.stefan_boltzmann_law)
print(test.stpv_power_density)

plt.plot(test.wavelength_array, test._solar_spectrum/1.4e9)
plt.plot(test.wavelength_array, test._atmospheric_transmissivity)
plt.show()




