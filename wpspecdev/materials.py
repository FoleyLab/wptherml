import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import os

path_and_file = os.path.realpath(__file__)
path = path_and_file[:-12]
class Materials():
    """ Compute the absorption, scattering, and extinction spectra of a sphere using Mie theory
    
    """

    def material_air(self, layer_number):
        self._refractive_index_array[:,layer_number] = np.ones(len(self.wavelength_array), dtype=complex) * 1.0

    def material_sio2(self, layer_number):
        if layer_number>0 and layer_number<(self.number_of_layers-1):
            # get path to the sio2 data file
            file_path = path + 'data/sio2_ir.txt'
            # now read sio2 data into a numpy array
            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index
            n_spline = InterpolatedUnivariateSpline(file_data[:,0], file_data[:,1], k=1)
            k_spline = InterpolatedUnivariateSpline(file_data[:,0], file_data[:,2], k=1)

            self._refractive_index_array[:,layer_number] = n_spline(self.wavelength_array) + 1j * k_spline(self.wavelength_array)
        #print()
    def material_tio2(self, layer_number):
        if layer_number>0 and layer_number<(self.number_of_layers-1):
            self._refractive_index_array[:,layer_number] = np.ones(len(self.wavelength_array), dtype=complex) * 2.4

        
        
        

