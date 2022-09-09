import wptherml
from matplotlib import pyplot as plt
import numpy as np

test_args = {
    "wavelength_list": [300e-9, 6000e-9, 1000],
    "Material_List": ["Air", "Al2O3", "SiO2_ir.txt", "TiO2", "Ag_JC.txt", "Al2O3", "W", "Air"],
    "Thickness_List": [0, 20e-9, 255e-9, 150e-9, 255e-9, 10e-9, 900e-9, 0],
    "temperature": 1700,
    "therml": True
}


sf = wptherml.SpectrumFactory()
test = sf.spectrum_factory('Tmm', test_args)


plt.plot(test.wavelength_array, test.thermal_emission_array, 'red')
plt.plot(test.wavelength_array, test.blackbody_spectrum, 'black')
plt.show()


