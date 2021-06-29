import wpspecdev
import numpy as np
from matplotlib import pyplot as plt

thickness = 100e-9
userargs = {
    'radius': 170e-9,
    'wavelength_list': [400e-9, 9000e-9, 500]
}
sf = wpspecdev.SpectrumFactory()
mt = sf.spectrum_factory('Mie', userargs)
mt.material_Ta2O5(1)

