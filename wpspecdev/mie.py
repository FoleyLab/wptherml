from .spectrum_driver import SpectrumDriver
import numpy as np

class MieDriver(SpectrumDriver):
    def __init__(self, radius):
        self.radius = radius
        print('Radius of the sphere is ', self.radius)

    def compute_spectrum(self):
        """ Compute the absorption, scattering, and extinction spectra of a sphere using Mie theory

             Attributes
             ----------
             radius : float
                 the radius of the sphere   
             wavelength_array : numpy array of floats
                 the array of wavelengths in meters over which you will compute the spectra
           
             _refractive_index_array : numpy array of complex floats
                 the array of refractive index values corresponding to wavelength_array

              Q_scat : numpy array of floats
                  the scattering efficiency as a function of wavelength

              Q_ext : numpy array of floats
                  the scattering efficiency as a function of wavelength

              Q_abs : numpy array of floats
                  the scattering efficiency as a function of wavelength

              Returns
              -------
              None
    
              Examples
              --------
              >>> fill_in_with_actual_example!
        """
        self.Q_scat = self.radius * 2
        self.Q_ext = self.radius * 3
        self.Q_abs = self.radius * 4
        return np.pi * self.radius
