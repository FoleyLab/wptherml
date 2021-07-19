import wpspecdev
from matplotlib import pyplot as plt
args = {  
'wavelength_list': [300e-9, 800e-9, 100],  
'sphere_material': "au",
'medium_material': "air",
'radius': 5e-9  
}  

sf = wpspecdev.SpectrumFactory()  
mt_5 = sf.spectrum_factory('Mie', args)
args['radius'] = 20e-9
mt_20 = sf.spectrum_factory('Mie', args)
args['radius'] = 40e-9
mt_40 = sf.spectrum_factory('Mie', args)
args['radius'] = 80e-9
mt_80 = sf.spectrum_factory('Mie', args)




plt.plot(mt_5.wavelength_array*1e9, mt_5.q_abs, label='r=5 nm')
plt.plot(mt_5.wavelength_array*1e9, mt_20.q_abs, label='r=20 nm')
plt.plot(mt_5.wavelength_array*1e9, mt_40.q_abs,label='r=40 nm')
plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorption efficiency")
plt.legend()
plt.savefig("absorption.png")
plt.show()



