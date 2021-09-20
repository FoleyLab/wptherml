import wpspecdev
from matplotlib import pyplot as plt
water_args = {  
'wavelength_list': [200e-9, 1100e-9, 901],  
'sphere_material': "pt",
'medium_material': "water",
'radius': 2e-9  
}  

air_args = {
'wavelength_list': [200e-9, 1100e-9, 901],
'sphere_material': "pt",
'medium_material': "air",
'radius': 2e-9  
} 

sf = wpspecdev.SpectrumFactory()  
pt_air = sf.spectrum_factory('Mie', air_args)
pt_water = sf.spectrum_factory('Mie', water_args)

for i in range(0, 901):
    print(pt_air.wavelength_array[900-i]*1e9, pt_air.q_ext[900-i], pt_water.q_ext[900-i])


plt.plot(pt_air.wavelength_array*1e9, pt_air.q_ext, 'red')
plt.plot(pt_water.wavelength_array*1e9, pt_water.q_ext, 'blue')
plt.xlabel("Wavelength (nm)")
plt.ylabel(" Extinction")
plt.legend()
plt.show()


#mt_5.compute_hamiltonian( )



