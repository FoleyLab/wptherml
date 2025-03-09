import wptherml
from matplotlib import pyplot as plt
import numpy as np
import time

# guess thickness for glass
d1 = 4500e-9 / (4 * 1.5)
# guess thickness for Si3N4
d2 = 4500e-9 / (4 * 2.04)

print(F"Trial thickness of glass layer is {d1:.3e} m")
print(F"Trial thickness of alumina layer is {d2:.3e} m")

test_args = {
    "wavelength_list": [300e-9, 6000e-9, 1000],
    "Material_List": ["Air",
                      "SiO2", "Si3N4", "SiO2", "Si3N4","SiO2", "Si3N4",
                      "SiO2", "Si3N4","SiO2", "Si3N4","SiO2", "Si3N4",
                      "SiO2", "Si3N4","SiO2", "Si3N4","SiO2", "Si3N4",
                      "Air"],
    "Thickness_List": [0,
                       d1, d2, d1, d2, d1, d2,
                       d1, d2, d1, d2, d1, d2,
                       d1, d2, d1, d2, d1, d2,
                       0],
    "reflective_window_wn" : [2000, 2400],
    "transmissive_window_nm" : [350, 700],
    "transmission_efficiency_weight" : 0.0,
    "reflection_efficiency_weight" : 0.5,
    "reflection_selectivity_weight" : 0.5,
 }

sf = wptherml.SpectrumFactory()


# create an instance of the DBR called test
test = sf.spectrum_factory('Tmm', test_args)


test.compute_selective_mirror_fom()

# compute the gradient of the foms - this is slower than computing the FOM itself
test.compute_selective_mirror_fom_gradient()


