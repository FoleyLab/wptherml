import wpspecdev
from matplotlib import pyplot as plt
args = {  
'wavelength_list': [300e-9, 800e-9, 500],  
'sphere_material': "tio2",
'medium_material': "air",
'radius': 500e-9  
}  

sf = wpspecdev.SpectrumFactory()  
mt_5 = sf.spectrum_factory('Mie', args)
args['radius'] = 600e-9
mt_20 = sf.spectrum_factory('Mie', args)
args['radius'] = 700e-9
mt_40 = sf.spectrum_factory('Mie', args)
args['radius'] = 800e-9
mt_80 = sf.spectrum_factory('Mie', args)




plt.plot(mt_5.wavelength_array*1e9, mt_5.q_scat, label='r=5 nm')
plt.plot(mt_5.wavelength_array*1e9, mt_20.q_scat, label='r=20 nm')
plt.plot(mt_5.wavelength_array*1e9, mt_40.q_scat,label='r=40 nm')
plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorption efficiency")
plt.legend()
plt.show()



