import numpy as np



class Materials():
    """ Compute the absorption, scattering, and extinction spectra of a sphere using Mie theory
    
    """
    def material_air(self, layer_number):
        self._refractive_index_array[:,layer_number] = np.ones(len(self.wavelength_array), dtype=complex) * 1.0

    def material_sio2(self, layer_number):
        if layer_number>0 and layer_number<(self.number_of_layers-1):
            self._refractive_index_array[:,layer_number] = np.ones(len(self.wavelength_array), dtype=complex) * 1.5

    def material_tio2(self, layer_number):
        if layer_number>0 and layer_number<(self.number_of_layers-1):
            self._refractive_index_array[:,layer_number] = np.ones(len(self.wavelength_array), dtype=complex) * 2.4

        
        
        

