import wpspecdev
import numpy as np
from matplotlib import pyplot as plt

thickness = 100e-9
userargs = {
    'wavelength_list': [400e-9, 800e-9, 10]
}
sf = wpspecdev.SpectrumFactory()
mt = sf.spectrum_factory('Tmm', userargs)



# before compute spectrum, mt.q_ext is not define
mt.compute_spectrum()
print(mt.reflectivity_array)




