from wptherml.wpml import multilayer
from matplotlib import pyplot as plt
import numpy as np


structure = {
        'Temperature': 300,
        ### actual materials the structure is made from
        ### values are stored in the attribute self.n
        'Material_List': ['Air', 'Al2O3', 'Air'],
        ### thickness of each layer... terminal layers must be set to zero
        ### values are stored in attribute self.d
        'Thickness_List': [0, 200e-9, 0],
         ### range of wavelengths optical properties will be calculated for
         ### values are stored in the array self.lam
        'Lambda_List': [400e-9, 800e-9, 1000]
        }

### create the instance called glass_slab
glass_slab = multilayer(structure)