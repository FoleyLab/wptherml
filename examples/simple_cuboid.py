import wptherml
from matplotlib import pyplot as plt
args = {  
'wavelength_list': [400e-9, 800e-9, 500],  
'material': "au",
'medium_material': "air",
'Lx': 80e-9,
'Ly': 80e-9,
'Lz': 80e-9,  
}  

sf = wptherml.SpectrumFactory()  
test = sf.spectrum_factory('Cuboid', args)

plt.plot(test.wavelength_array*1e9, test.q_ext, 'red')
#plt.plot(test.wavelength_array*1e9, test.q_scat, 'blue')
plt.xlabel("Wavelength (nm)")
plt.ylabel(" efficiency")
plt.legend()
plt.show()




