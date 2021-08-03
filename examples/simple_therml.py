import wpspecdev
from matplotlib import pyplot as plt
args = {  
'wavelength_list': [300e-9, 800e-9, 500],  
'material_list': ["Air", "W", "Air"],
'thickness_list': [0,  100e-9, 0]
}  

sf = wpspecdev.SpectrumFactory()  
test = sf.spectrum_factory('Therml', args)

#plt.plot(test.wavelength_array*1e9, test.q_abs, 'red')
#plt.plot(test.wavelength_array*1e9, test.q_scat, 'blue')
#plt.xlabel("Wavelength (nm)")
#plt.ylabel(" efficiency")
#plt.legend()
#plt.show()


#mt_5.compute_hamiltonian( )



