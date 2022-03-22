# validated against this google colab notebook with v1 of wptherml
# https://colab.research.google.com/drive/13YwoHzJwmNpElhZlYX8FphDImvaXDvQ4?usp=sharing 
import wpspecdev
from matplotlib import pyplot as plt
import numpy as np

test_args = {
    "wavelength_list": [300e-9, 6000e-9, 1000],
    "Material_List": ["Air", "TiN", "Air"],
    "Thickness_List" : [0, 400e-9, 0],
    "temperature": 1700,
    "therml": True
}

sf = wpspecdev.SpectrumFactory()  
test = sf.spectrum_factory('Tmm', test_args)

print(test.lambda_bandgap)
print("Printing STPV Figures of Merit")
print(test.stpv_power_density)
print(test.stpv_spectral_efficiency)

plt.plot(test.wavelength_array, test.thermal_emission_array, 'red')
plt.plot(test.wavelength_array, test.blackbody_spectrum, 'black')
plt.show()

test.compute_stpv()
test.compute_stpv_gradient()
print(test.power_density_gradient)
print(test.stpv_spectral_efficiency_gradient)