import wptherml
from matplotlib import pyplot as plt
import numpy as np

test_args = {

    # range of wavelengths in meters to compute tmm quantities
    "wavelength_list": [400e-9, 800e-9, 100], 
    # specify the materials you want to use... first and last should be something
    # completely non-absorbing, "Air" is a good choice
    "material_list": ["Air", "SiO2", "Au", "Air"],
    # thickness of each layer in meters matching the order of the materials... terminal layers
    # are infinite, so of course we say "0"
    "thickness_list": [0, 200e-9, 10e-9, 0], 
}

# the top-level driver class is called "SpectrumFactory" - create an instance of it
sf = wptherml.SpectrumFactory()  

# the spectrum factory can be used to create an instance of a multilayer
# that will automatically have attributes related to the materials and geometry
# of the multilayer, along with the reflectivity, transmissivity, and absorptivity/emissivity spectra
ts = sf.spectrum_factory("Tmm", test_args)

# now you can access these attributes, for example you can plot 
# the reflectivity, transmissivity, emissivity as follows
plt.plot(ts.wavelength_array, ts.reflectivity_array)
plt.plot(ts.wavelength_array, ts.transmissivity_array)
plt.plot(ts.wavelength_array, ts.emissivity_array)
plt.show()




