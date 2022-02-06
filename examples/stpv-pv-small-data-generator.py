import wpspecdev
from matplotlib import pyplot as plt
import numpy as np

# define a test structure
test_args = {
    "therml": True,
    'TeMperature' : 440,
    "wavelength_list": [300e-9, 20300e-9, 1001],
    "material_list": ['Air', 'SiO2', 'Al', 'SiO2', 'Ta2O5', 'polystyrene', 'SiO2', 'Al', 'SiO2', 'Ta2O5', 'Air'], 
    "thickness_list": [0, 2.6422605026007024e-07, 1.5e-09, 2.6422605026007024e-07, 3.733648095151694e-07,  0.001, 3.8014616788692957e-07, 1.5e-09, 3.8014616788692957e-07, 5.311796118517261e-07, 0]
}

import time
start = time.time()

sf = wpspecdev.SpectrumFactory()  
# create instance of class
ts = sf.spectrum_factory("Tmm", test_args)

# want to create a step function so that we can only integrate
# a subinterval of the entire reflectivity function up to 760 nm
target_pv = 760e-9 * np.ones_like(ts.wavelength_array)
stop_idx = np.argmin(np.abs( target_pv - ts.wavelength_array) )

# step function to cut off at \lambda_bg of the PV
step_pv = np.zeros_like(ts.wavelength_array)
step_pv[:stop_idx] = 1.

# do the same for the to create a step function between 1000 and 3500 nm
target_min = 1000e-9 * np.ones_like(ts.wavelength_array)
target_max = 3500e-9 * np.ones_like(ts.wavelength_array)

min_idx = np.argmin(np.abs( target_min - ts.wavelength_array))
max_idx = np.argmin(np.abs( target_max - ts.wavelength_array))

step_swir = np.zeros_like(ts.wavelength_array)
step_swir[min_idx:max_idx] = 1.


### Integrate the above-gap solar flux transmitted through optic
AM_transmit_power = np.trapz(ts._solar_spectrum * step_pv *  ts.transmissivity_array, ts.wavelength_array)
### Integrate the thermal emission of the optic in the SWIR
SWIR_emit_power = np.trapz(ts.thermal_emission_array * step_swir, ts.wavelength_array)

### Integrate the total above-gap solar flux
AM_above_gap = np.trapz(ts._solar_spectrum * step_pv, ts.wavelength_array)
### Integrate the total BB emission at 440 K in the SWIR
SWIR_power = np.trapz(ts.blackbody_spectrum * step_swir, ts.wavelength_array)

end = time.time()
print("time required")
print(end-start)
print(SWIR_emit_power,  " W / s / m^2 emitted")
print(AM_transmit_power," W / s / m^2 transmitted")

print(SWIR_power,  " W / s / m^2 emitted")
print(AM_above_gap," W / s / m^2 transmitted")


import random
def update_multilayer(SP, AP):
    d3 = random.uniform(1, 700)
    d4 = random.uniform(1, 700)
    d8 = random.uniform(1, 700)
    d9 = random.uniform(1, 700)

    ts.thickness_array[3] = d3 * 1e-9
    ts.thickness_array[4] = d4 * 1e-9
    ts.thickness_array[8] = d8 * 1e-9
    ts.thickness_array[9] = d9 * 1e-9

    ts.compute_spectrum()
    ts._compute_therml_spectrum(ts.wavelength_array, ts.emissivity_array)
    AM_transmit_power = np.trapz(ts._solar_spectrum * step_pv *  ts.transmissivity_array, ts.wavelength_array)
    ### Integrate the thermal emission of the optic in the SWIR
    SWIR_emit_power = np.trapz(ts.thermal_emission_array * step_swir, ts.wavelength_array)
    d = np.array([d3, d4, d8, d9])
    return d/700, AM_transmit_power/AP, SWIR_emit_power/SP
    
dim = 5000

f = open('stpv_pv_5kd.csv', 'w')
for i in range(0, dim):
    d, T_AM, E_SWIR = update_multilayer(SWIR_power, AM_above_gap)
    #print(d1, d2, d3, d4, pv_2l.conversion_efficiency_val)
    pr_str = str(d[0]) + ", " + str(d[1]) + ", " + str(d[2]) + ", "
    pr_str += str(d[3]) + ", " + str(T_AM) + ", " + str(E_SWIR)
    pr_str += "\n"
    f.write(pr_str)
    
f.close()
