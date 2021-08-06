import wpspecdev
from matplotlib import pyplot as plt
args = {  
'wavelength_list': [300e-9, 800e-9, 500],  
'sphere_material': "Ag",
'medium_material': "air",
'radius': 10e-9  
}  

sf = wpspecdev.SpectrumFactory()  
test = sf.spectrum_factory('Mie', args)

plt.plot(test.wavelength_array*1e9, test.q_abs, 'red')
plt.plot(test.wavelength_array*1e9, test.q_scat, 'blue')
plt.xlabel("Wavelength (nm)")
plt.ylabel(" efficiency")
plt.legend()
plt.show()


#mt_5.compute_hamiltonian( )



