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
    "wavelength_list": [300e-9, 20300e-9, 5001],
    "material_list": ['Air', 'SiO2', 'Al', 'SiO2', 'Ta2O5', 'SiO2', 'Ta2O5', 'SiO2', 'Ta2O5', 'SiO2', 'polystyrene', 'SiO2', 'Al', 'SiO2', 'Ta2O5', 'SiO2', 'Ta2O5', 'SiO2', 'Ta2O5', 'SiO2', 'Air'], 
  "thickness_list": [0, 2.6422605026007024e-07, 1.5e-09, 2.6422605026007024e-07, 3.733648095151694e-07, 5.284521005201405e-07, 3.733648095151694e-07, 5.284521005201405e-07, 3.733648095151694e-07, 0.0001, 0.001, 3.8014616788692957e-07, 1.5e-09, 3.8014616788692957e-07, 5.311796118517261e-07, 7.602923357738591e-07, 5.311796118517261e-07, 7.602923357738591e-07, 5.311796118517261e-07, 0.0001, 0]
}
import time
start = time.time()
sf = wpspecdev.SpectrumFactory()  
# create instance of class
ts = sf.spectrum_factory("Tmm", test_args)

#plt.plot(ts.wavelength_array, ts._solar_spetrum) 
#plt.show()

h = 6.626e-34
c = 299792458.

### Get the solar flux spectrum (number of photons per second per wavelength per meter squared)
AMflux = ts._solar_spetrum * ts.wavelength_array / (h * c)

# want to create a step function so that we can only integrate
# a subinterval of the entire reflectivity function up to 760 nm
target = 760e-9 * np.ones_like(ts.wavelength_array)
stop_idx = np.argmin(np.abs( target - ts.wavelength_array) )

# step function
step = np.zeros_like(ts.wavelength_array)
step[:stop_idx] = 1

### Integrate the above-gap solar flux transmitted through optic
AM_transmit_power = np.trapz(ts._solar_spetrum * step *  (1-ts.reflectivity_array), ts.wavelength_array)
end = time.time()
print("time required")
print(end-start)
AM_transmit_flux = np.trapz(AMflux * step *  ts.transmissivity_array, ts.wavelength_array)
print(AM_transmit_flux," photons / s / m^2 transmitted")
print(AM_transmit_power," W / s / m^2 transmitted")


