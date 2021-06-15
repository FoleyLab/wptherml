import wpspecdev
import numpy as np

thickness = 100e-9

sf = wpspecdev.SpectrumFactory()
mt = sf.spectrum_factory('Tmm', thickness)
#mt.material_sio2(1)
#print(mt._medium_refractive_index)
