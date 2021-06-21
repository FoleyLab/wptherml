import wpspecdev
import numpy as np
from matplotlib import pyplot as plt

thickness = 100e-9
userargs = {
    'radius': 170e-9,
    'wavelength_list': [100e-9, 800e-9, 500]
}
sf = wpspecdev.SpectrumFactory()
mt_1 = sf.spectrum_factory('Mie', userargs)
mt_1.compute_spectrum()

userargs2 = {
    'radius': 100e-9,
    'wavelength_list': [100e-9, 800e-9, 500]
}
mt_2 = sf.spectrum_factory('Mie', userargs2)
mt_2.compute_spectrum()

plt.plot(mt_1.wavelength_array*1e9, mt_1.q_ext)
plt.plot(mt_2.wavelength_array*1e9, mt_2.q_ext)
plt.show()
#print(mt._size_factor_array)
#mt.material_TiO2(1)
#print(mt._refractive_index_array[:,1])
#materials_test = wpspecdev.Materials()

