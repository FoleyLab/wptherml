import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import os

path_and_file = os.path.realpath(__file__)
path = path_and_file[:-12]
class Materials():
    """ Compute the absorption, scattering, and extinction spectra of a sphere using Mie theory
    
    """
    def _create_test_multilayer(self, central_wavelength):
        """
        Simple method to create a 3-entry array of wavelengths as follows:
        [central_wavelength-1e-9 m, central_wavelength m, central_wavelength+1e-9 m]
        and dummy _refractive_index_array that can be filled in 
        with actual materials at the wavelength arrays.
        This is simply meant to enable unit testing for desired wavelengths of the 
        various materials methods
        """
        self.wavelength_array = np.array([central_wavelength-1e-9, central_wavelength, central_wavelength+1e-9])
        self.number_of_wavelengths = 3 
        self.number_of_layers = 3
        self._refractive_index_array = np.reshape(np.tile(np.array([1+0j, 1+0j, 1+0j]), self.number_of_wavelengths), 
                                                  (self.number_of_wavelengths, self.number_of_layers)) 

    def material_Air(self, layer_number):
        self._refractive_index_array[:,layer_number] = np.ones(len(self.wavelength_array), dtype=complex) * 1.0

    def material_SiO2(self, layer_number):
        if layer_number>0 and layer_number<(self.number_of_layers-1):
            # get path to the sio2 data file
            file_path = path + 'data/SiO2_ir.txt'
            # now read sio2 data into a numpy array
            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index
            n_spline = InterpolatedUnivariateSpline(file_data[:,0], file_data[:,1], k=1)
            k_spline = InterpolatedUnivariateSpline(file_data[:,0], file_data[:,2], k=1)

            self._refractive_index_array[:,layer_number] = n_spline(self.wavelength_array) + 1j * k_spline(self.wavelength_array)

    def material_TiO2(self, layer_number):
        if layer_number>0 and layer_number<(self.number_of_layers-1):
            # get path to the tio2 data file
            file_path = path + 'data/TiO2_Siefke.txt'
            # now read TiO2 data into a numpy array
            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index
            n_spline = InterpolatedUnivariateSpline(file_data[:,0], file_data[:,1], k=1)
            k_spline = InterpolatedUnivariateSpline(file_data[:,0], file_data[:,2], k=1)

            self._refractive_index_array[:,layer_number] = n_spline(self.wavelength_array) + 1j * k_spline(self.wavelength_array)

    def material_Ta2O5(self, layer_number):
        if layer_number>0 and layer_number<(self.number_of_layers-1):
            # get path to the tio2 data file
            file_path = path + 'data/Ta2O5_Bright.txt'
            # now read TiO2 data into a numpy array
            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index
            n_spline = InterpolatedUnivariateSpline(file_data[:,0], file_data[:,1], k=1)
            k_spline = InterpolatedUnivariateSpline(file_data[:,0], file_data[:,2], k=1)

            self._refractive_index_array[:,layer_number] = n_spline(self.wavelength_array) + 1j * k_spline(self.wavelength_array)
        
    def material_TiN(self, layer_number):
        if layer_number>0 and layer_number<(self.number_of_layers-1):
            # get path to the tio2 data file
            file_path = path + 'data/TiN_ellipsometry_data.txt'
            # now read TiO2 data into a numpy array
            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index
            n_spline = InterpolatedUnivariateSpline(file_data[:,0], file_data[:,1], k=1)
            k_spline = InterpolatedUnivariateSpline(file_data[:,0], file_data[:,2], k=1)

            self._refractive_index_array[:,layer_number] = n_spline(self.wavelength_array) + 1j * k_spline(self.wavelength_array)   

    def material_Al(self, layer_number):
        if layer_number>0 and layer_number<(self.number_of_layers-1):
            # get path to the Al data file
            file_path = path + 'data/Al.txt'
            # now read Al data into a numpy array
            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index
            n_spline = InterpolatedUnivariateSpline(file_data[:,0], file_data[:,1], k=1)
            k_spline = InterpolatedUnivariateSpline(file_data[:,0], file_data[:,2], k=1)

            self._refractive_index_array[:,layer_number] = n_spline(self.wavelength_array) + 1j * k_spline(self.wavelength_array)      

    def material_W(self, layer_number):
        if layer_number>0 and layer_number<(self.number_of_layers-1):
            # get path to the W data file
            file_path = path + 'data/W.txt'
            # now read W data into a numpy array
            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index
            n_spline = InterpolatedUnivariateSpline(file_data[:,0], file_data[:,1], k=1)
            k_spline = InterpolatedUnivariateSpline(file_data[:,0], file_data[:,2], k=1)

            self._refractive_index_array[:,layer_number] = n_spline(self.wavelength_array) + 1j * k_spline(self.wavelength_array)      

    def material_AlN(self, layer_number):
        if layer_number>0 and layer_number<(self.number_of_layers-1):
            # get path to the AlN data file
            file_path = path + 'data/AlN.txt'
            # now read AlN data into a numpy array
            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index
            n_spline = InterpolatedUnivariateSpline(file_data[:,0], file_data[:,1], k=1)
            k_spline = InterpolatedUnivariateSpline(file_data[:,0], file_data[:,2], k=1)
            self._refractive_index_array[:,layer_number] = n_spline(self.wavelength_array) + 1j * k_spline(self.wavelength_array)   

    def material_HfO2(self, layer_number):
        if layer_number>0 and layer_number<(self.number_of_layers-1):
            self._refractive_index_array[:,layer_number] = np.ones(len(self.wavelength_array), dtype=complex) * 2.4
            # get path to the HfO2 data file
            file_path = path + 'data/HfO2.txt'
            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index
            n_spline = InterpolatedUnivariateSpline(file_data[:,0], file_data[:,1], k=1)
            k_spline = InterpolatedUnivariateSpline(file_data[:,0], file_data[:,2], k=1)
            self._refractive_index_array[:,layer_number] = n_spline(self.wavelength_array) + 1j * k_spline(self.wavelength_array)   

