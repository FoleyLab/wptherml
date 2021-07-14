import wpspecdev
import numpy as np
from matplotlib import pyplot as plt

thickness = 100e-9
userargs = {
    'wavelength_list': [400e-9, 3000e-9, 500],
    'material_list': ["Air", "TiO2", "SiO2", "Air"],
    'thickness_list': [0, 200e-9, 100e-9, 0]
}
sf = wpspecdev.SpectrumFactory()
mt = sf.spectrum_factory('Tmm', userargs)
mt.set_refractive_indicex_array()





# before compute spectrum, mt.q_ext is not define
mt.compute_spectrum()

plt.plot(mt.wavelength_array, mt.reflectivity_array)
plt.show()




