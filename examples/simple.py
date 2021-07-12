import wpspecdev
import numpy as np
from matplotlib import pyplot as plt

thickness = 100e-9
userargs = {
    'wavelength_list': [450e-9, 9050e-9, 10]
}
sf = wpspecdev.SpectrumFactory()
mt = sf.spectrum_factory('Tmm', userargs)

mt.compute_spectrum()



