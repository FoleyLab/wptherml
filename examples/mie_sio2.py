import wpspecdev
from matplotlib import pyplot as plt

sio2_112 = {  
'wavelength_list': [200e-9, 1100e-9, 901],  
'sphere_material': "SiO2",
'medium_material': "air",
'radius': 112e-9/2.  
}  

sio2_173 = sio2_112.copy()
sio2_262 = sio2_112.copy() 
sio2_353 = sio2_112.copy()
sio2_441 = sio2_112.copy()
sio2_561 = sio2_112.copy()
sio2_680 = sio2_112.copy()
sio2_864 = sio2_112.copy()
sio2_1089 = sio2_112.copy() 
sio2_1300 = sio2_112.copy()
sio2_1500 = sio2_112.copy()

sio2_173["radius"] = 173e-9/2.
sio2_262["radius"] = 262e-9/2.
sio2_353["radius"] = 353e-9/2.
sio2_441["radius"] = 441e-9/2.
sio2_561["radius"] = 561e-9/2.
sio2_680["radius"] = 680e-9/2.
sio2_864["radius"] = 864e-9/2.
sio2_1089["radius"] = 1089e-9/2.
sio2_1300["radius"] = 1300e-9/2.
sio2_1500["radius"] = 1500e-9/2.


sf = wpspecdev.SpectrumFactory()  
sio2_112_sp = sf.spectrum_factory('Mie', sio2_112)
sio2_173_sp = sf.spectrum_factory('Mie', sio2_173)
sio2_262_sp = sf.spectrum_factory('Mie', sio2_262)
sio2_353_sp = sf.spectrum_factory('Mie', sio2_353)
sio2_441_sp = sf.spectrum_factory('Mie', sio2_441)
sio2_561_sp = sf.spectrum_factory('Mie', sio2_561)
sio2_680_sp = sf.spectrum_factory('Mie', sio2_680)
sio2_864_sp = sf.spectrum_factory('Mie', sio2_864)
sio2_1089_sp = sf.spectrum_factory('Mie', sio2_1089)
sio2_1300_sp = sf.spectrum_factory('Mie', sio2_1300)
sio2_1500_sp = sf.spectrum_factory('Mie', sio2_1500)


for i in range(0, 901):
    print(sio2_112_sp.wavelength_array[900-i]*1e9, 
          sio2_112_sp.q_ext[900-i],
          sio2_173_sp.q_ext[900-i],
          sio2_262_sp.q_ext[900-i],
          sio2_353_sp.q_ext[900-i],
          sio2_441_sp.q_ext[900-i],
          sio2_561_sp.q_ext[900-i],
          sio2_680_sp.q_ext[900-i],
          sio2_864_sp.q_ext[900-i],
          sio2_1089_sp.q_ext[900-i],
          sio2_1300_sp.q_ext[900-i],
          sio2_1500_sp.q_ext[900-i])

#plt.plot(pt_air.wavelength_array*1e9, pt_air.q_ext, 'red')
#plt.plot(pt_water.wavelength_array*1e9, pt_water.q_ext, 'blue')
#plt.xlabel("Wavelength (nm)")
#plt.ylabel(" Extinction")
#plt.legend()
#plt.show()


#mt_5.compute_hamiltonian( )



