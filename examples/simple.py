import wpspecdev
import numpy as np

thickness = 100e-9

#sf = wpspecdev.SpectrumFactory()
#mt = sf.spectrum_factory('Tmm', thickness)
#mt.material_TiO2(1)
#print(mt._refractive_index_array[:,1])
materials_test = wpspecdev.Materials()

materials_test._create_test_multilayer(central_wavelength=636e-9)
print(materials_test.wavelength_array)