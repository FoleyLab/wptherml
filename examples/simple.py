import wpspecdev
import numpy as np
from matplotlib import pyplot as plt

thickness = 100e-9
userargs = {
    'wavelength_list': [500e-9, 502e-9, 3],
    'material_list': ["Air", "TiO2", "SiO2", "Ag", "Au", "Pt", "AlN", "Al2O3", "Air"],
    'thickness_list': [0, 200e-9, 100e-9, 5e-9, 6e-9, 7e-9, 100e-9, 201e-9, 0]
}
sf = wpspecdev.SpectrumFactory()
mt = sf.spectrum_factory('Tmm', userargs)

#print(mt._refractive_index_array[1,:])
#print(mt.wavelength_array[1])





# before compute spectrum, mt.q_ext is not define

#print(mt.reflectivity_array)






