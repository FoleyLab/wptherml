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
    "wavelength_list": [600e-9, 602e-9, 3],
    "material_list": [
    "Air",
    "SiO2",
    "Au", 
    "Air",
    ],
    "thickness_list": [0, 200e-9, 10e-9, 0],
}
sf = wpspecdev.SpectrumFactory()  
# create instance of class
ts = sf.spectrum_factory("Tmm", test_args)
print(ts._kz_array[0,:])
ts._refractive_index_array[0,1] = 1.47779002+0.j
ts._refractive_index_array[0,2] = 0.24463382+3.085112j
ts.compute_spectrum()
print(ts._kz_array[0,:])
ts.compute_spectrum_gradient()
print("grad")
print(ts.reflectivity_gradient_array[0,:])


#test = sf.spectrum_factory('Tmm', test_1_args)
#test.render_color("Tritanopia", colorblindness="Tritanopia")
#test.render_color("Deuteranopia", colorblindness="Deuteranopia")
#test.render_color("Protanopia", colorblindness="Protanopia")
#test.render_color("Full Color Vision", colorblindness="False")

 # Protanopia Deuteranopia Tritanopia
#print(test.reflectivity_array[0])
#mt_5.compute_hamiltonian( )



