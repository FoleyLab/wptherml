import wpspecdev
from matplotlib import pyplot as plt
import numpy as np
#test_1_args = {
#    "wavelength_list": [400e-9, 800e-9, 1000],
#    "material_list": [
#        "Air",
#        "Au",
#        "Air",
#    ],
#    "thickness_list": [0, 200e-9, 0],
#}

test_args = {
    "wavelength_list": [300e-9, 3000e-9, 500],
    "material_list": ['Air', 'SiO2', 'Al', 'SiO2', 'Ta2O5', 'SiO2', 'Ta2O5', 'SiO2', 'Ta2O5', 'SiO2', 'polystyrene', 'SiO2', 'Al', 'SiO2', 'Ta2O5', 'SiO2', 'Ta2O5', 'SiO2', 'Ta2O5', 'SiO2', 'Air'], 
  "thickness_list": [0, 2.6422605026007024e-07, 1.5e-09, 2.6422605026007024e-07, 3.733648095151694e-07, 5.284521005201405e-07, 3.733648095151694e-07, 5.284521005201405e-07, 3.733648095151694e-07, 0.0001, 0.001, 3.8014616788692957e-07, 1.5e-09, 3.8014616788692957e-07, 5.311796118517261e-07, 7.602923357738591e-07, 5.311796118517261e-07, 7.602923357738591e-07, 5.311796118517261e-07, 0.0001, 0]
}

sf = wpspecdev.SpectrumFactory()  
# create instance of class
ts = sf.spectrum_factory("Tmm", test_args)

plt.plot(ts.wavelength_array, ts._solar_spetrum) 
plt.show()



