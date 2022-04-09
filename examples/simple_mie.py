import wptherml
from matplotlib import pyplot as plt
args = {  
'wavelength_list': [200e-9, 1100e-9, 500],  
'sphere_material': "SiO2",
'medium_material': "air",
'radius': 200e-9  
}  

sf = wptherml.SpectrumFactory()  
test = sf.spectrum_factory('Mie', args)

plt.plot(test.wavelength_array*1e9, test.q_abs, 'red')
plt.plot(test.wavelength_array*1e9, test.q_scat, 'blue')
plt.xlabel("Wavelength (nm)")
plt.ylabel(" efficiency")
plt.legend()
plt.show()




