import wptherml
from matplotlib import pyplot as plt
import numpy as np

# guess thickness for glass
d1 = 5000e-9 / (4 * 1.5)
# guess thickness for Al2O3
d2 = 5000e-9 / (4 * 1.85)

test_args = {
    "wavelength_list": [300e-9, 6000e-9, 1000],
    "Material_List": ["Air","SiO2", "Al2O3", "SiO2", "ZrO2","SiO2", "Al2O3", "SiO2", "Al2O3","SiO2", "Al2O3", "SiO2", "Al2O3","SiO2", "Al2O3","SiO2", "Al2O3", "SiO2", "Al2O3","SiO2", "Al2O3", "SiO2", "Al2O3" , "Air"],
    "Thickness_List": [0,d1, d2, d1, d2,d1, d2, d1, d2, d1, d2, d1, d2, d1, d2,d1, d2, d1, d2, d1, d2, d1, d2, 0],
    "reflective_window_wn" : [2000, 2400],
    "transmissive_window_nm" : [350, 700],
 }
# the top-level driver class is called "SpectrumFactory" - create an instance of it
sf = wptherml.SpectrumFactory()  

# the spectrum factory can be used to create an instance of a multilayer
# that will automatically have attributes related to the materials and geometry
# of the multilayer, along with the reflectivity, transmissivity, and absorptivity/emissivity spectra
ts = sf.spectrum_factory("Opt", test_args)

