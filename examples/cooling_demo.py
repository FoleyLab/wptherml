# validated against this google colab notebook with v1 of wptherml
# https://colab.research.google.com/drive/13YwoHzJwmNpElhZlYX8FphDImvaXDvQ4?usp=sharing 
import wpspecdev
from matplotlib import pyplot as plt
import numpy as np

#test_args = {
### Define structure!
test_args = {

        'Material_List': ['Air', 'SiO2', 'HfO2', 'SiO2', 'HfO2', 'SiO2', 'HfO2', 'SiO2', 'Ag', 'Air'],
        'Thickness_List': [0, 230e-9, 485e-9, 688e-9, 13e-9, 73e-9, 34e-9, 54e-9, 200e-9, 0],
        'Wavelength_List': [300e-9, 20000e-9, 5000],
        'Cooling': True,
}
     
sf = wpspecdev.SpectrumFactory()  
test = sf.spectrum_factory('Tmm', test_args)

#print(test.lambda_bandgap)
#print("Printing STPV Figures of Merit")
#print(test.stpv_power_density)
#print(test.stpv_spectral_efficiency)

#plt.plot(test.wavelength_array, test.thermal_emission_array, 'red')
plt.plot(test.wavelength_array, test.emissivity_array, 'black')
plt.plot(test.wavelength_array, test._solar_spectrum/(1.4*1e9), 'blue')
#plt.plot(test.wavelength_array, test._atmospheric_transmissivity, 'cyan')
plt.xlim(300e-9,2000e-9)
plt.show()

plt.plot(test.wavelength_array*1e9, test._atmospheric_transmissivity, 'cyan', label='Atmospheric Transparency')
plt.plot(test.wavelength_array*1e9, test.emissivity_array, 'red', label='Emissivity')
### Uncomment the next line if you want to plot the transmissivity of
### the multilayer in the IR
#plt.plot(cool_ml.lambda_array*1e9, cool_ml.transmissivity_array, 'green', label='Transmissivity')
plt.xlabel("Wavelength (nm)")
plt.ylabel("Arb. units")
plt.legend(loc = 'best')
plt.xlim(2500,20000)
plt.show()

