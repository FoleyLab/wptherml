import wpspecdev
from matplotlib import pyplot as plt
import numpy as np
args = {  
'wavelength_list': [400e-9, 7000e-9, 4000],  
'material_list': ["Air", "TiN", "Air"],
'thickness_list': [0,  400e-9, 0],
'TeMperature' : 5000, 
'therml': True
}  

sf = wpspecdev.SpectrumFactory()  
test = sf.spectrum_factory('Tmm', args)

print(test.blackbody_power_density)
print(test.stefan_boltzmann_law)
print(test.stpv_power_density)
#idx = np.abs(test.wavelength_array-test.lambda_bandgap).argmin()
#print(idx)

#test._compute_stpv_power_density()

#plt.plot(test.wavelength_array*1e9, test.q_abs, 'red')
#plt.plot(test.wavelength_array*1e9, test.q_scat, 'blue')
#plt.xlabel("Wavelength (nm)")
#plt.ylabel(" efficiency")
#plt.legend()
#plt.show()


#mt_5.compute_hamiltonian( )



