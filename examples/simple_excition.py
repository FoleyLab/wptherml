import wptherml
from matplotlib import pyplot as plt
import numpy as np

args = {  
'exciton_energy': 1.5,
'number_of_monomers' : 10,
'displacement_between_monomers' : np.array([1, 0, 0]), 
'transition_dipole_moment' : np.array([0, 0, 0.5]) 
}  

sf = wptherml.SpectrumFactory()  
new_instance = sf.spectrum_factory('Frenkel', args)

print(new_instance._compute_H0_element(1, 0) )
print("coords")
print(new_instance.coords)
print("transition dipole moment")
print(new_instance.transition_dipole_moment)




