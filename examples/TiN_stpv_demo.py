# validated against this google colab notebook with v1 of wptherml
# https://colab.research.google.com/drive/13YwoHzJwmNpElhZlYX8FphDImvaXDvQ4?usp=sharing 
import wpspecdev
from matplotlib import pyplot as plt
import numpy as np

#test_args = {
#    "wavelength_list": [300e-9, 6000e-9, 1000],
#    "Material_List": ["Air", "TiN", "Air"],
#    "Thickness_List" : [0, 400e-9, 0],
#    "temperature": 1700,
#    "therml": True, 
#    "cooling": True,
#}
test_args = {
    "wavelength_list": [300e-9, 60000e-9, 5000],
    "material_list": ["Air", "TiN", "Air"],
    "thickness_list": [0, 230e-9, 0],
    "temperature": 300,
    "cooling": True,
}
sf = wpspecdev.SpectrumFactory()  
test = sf.spectrum_factory('Tmm', test_args)

#print(test.lambda_bandgap)
#print("Printing STPV Figures of Merit")
#print(test.stpv_power_density)
#print(test.stpv_spectral_efficiency)

#plt.plot(test.wavelength_array, test.thermal_emission_array, 'red')
#plt.plot(test.wavelength_array, test.blackbody_spectrum, 'black')
#plt.show()

#test.compute_stpv()
#test.compute_stpv_gradient()
#print(test.power_density_gradient)
#print(test.stpv_spectral_efficiency_gradient)

test.compute_cooling_gradient()
print("printing the radiated solar power gradient")
print(test.solar_radiated_power_gradient)
<<<<<<< HEAD

# define a displacement in thickness of SiO2
_delta_d_sio2 = 1e-9

test.incident_angle = test.solar_angle
# get forward solar power
test.thickness_array[1] += _delta_d_sio2
test.compute_cooling()

_solar_power_f = test.solar_radiated_power
_emitted_power_f = test.thermal_radiated_power
_atmospheric_power_f = test.atmospheric_radiated_power
print("PRITING FORWARD INFO")
print("P_sun",_solar_power_f)
print("P_rad",_emitted_power_f)
print("P_atm",_atmospheric_power_f)

# get backward solar power
test.thickness_array[1] -= 2 * _delta_d_sio2
test.compute_cooling()
_solar_power_b = test.solar_radiated_power
_emitted_power_b = test.thermal_radiated_power
_atmospheric_power_b = test.atmospheric_radiated_power
print("PRITING BACKWARD INFO")
print("P_sun",_solar_power_b)
print("P_rad",_emitted_power_b)
print("P_atm",_atmospheric_power_b)


_numeric_solar_power_gradient = (_solar_power_f-_solar_power_b)/ (2 * _delta_d_sio2)
_numeric_emitted_power_gradient = (_emitted_power_f-_emitted_power_b) / (2 * _delta_d_sio2)
_numeric_atmospheric_power_gradient = (_atmospheric_power_f-_atmospheric_power_b) / (2 * _delta_d_sio2)

print("PRINTING NUMERIC INFO")
print("P_sun'",_numeric_solar_power_gradient)
print("P_rad'",_numeric_emitted_power_gradient )
print("P_atm'", _numeric_atmospheric_power_gradient)

print("PRINTING ANALYTIC INFO")
print("P_sun'",test.solar_radiated_power_gradient)
print("P_rad'",test.thermal_radiated_power_gradient )
print("P_atm'",test.atmospheric_radiated_power_gradient)
=======
>>>>>>> c511dc19dd6330ca907db47f82ccff8d78d5ede5
