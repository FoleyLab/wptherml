import wptherml
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
    "wavelength_list": [400e-9, 800e-9, 100],
    "material_list": [
    "Air",
    "SiO2",
    "Au", 
    "Air",
    ],
    "thickness_list": [0, 200e-9, 10e-9, 0],
}
sf = wptherml.SpectrumFactory()  
# create instance of class
ts = sf.spectrum_factory("Tmm", test_args)
plt.plot(ts.wavelength_array, ts.reflectivity_array)
plt.show()


#test = sf.spectrum_factory('Tmm', test_1_args)
#test.render_color("Tritanopia", colorblindness="Tritanopia")
#test.render_color("Deuteranopia", colorblindness="Deuteranopia")
#test.render_color("Protanopia", colorblindness="Protanopia")
#test.render_color("Full Color Vision", colorblindness="False")

 # Protanopia Deuteranopia Tritanopia
#print(test.reflectivity_array[0])



