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
                
            number_of_wavelengths : int
                the number of wavelengths over which the cross sections / efficiencies will be computed
                
            wavelength_array : 1 x number_of_wavelengths numpy array of floats
                the array of wavelengths in meters over which you will compute the spectra
                
            _z_array : 1 x number_of_wavelengths numpy array of complex floats
                size factor of the sphere
           
            _refractive_index_array : 1 x number_of_wavelengths numpy array of complex floats
                the array of refractive index values corresponding to wavelength_array

            Q_scat : numpy array of floats
                the scattering efficiency as a function of wavelength

            Q_ext : numpy array of floats
                the extenction efficiency as a function of wavelength

            Q_abs : 1 x number_of_wavelengths numpy array of floats
                the absorption efficiency as a function of wavelength
                 
            C_scat : numpy array of floats
                the scattering cross section as a function of wavelength

            C_ext : numpy array of floats
                the extinction cross section as a function of wavelength

            C_abs : 1 x number_of_wavelengths numpy array of floats
                the absorption efficiency as a function of wavelength
                 
            _max_coefficient_n : int
                the maximum coefficient to be computed in the Mie expansion
                
            _n_array : 1 x _max_coefficient_n array of ints
                array of indices for the terms in the Mie expansion
                 
            _an : _max_coefficient x number_of_wavelengths numpy array of complex floats
                the array of a coefficients in the Mie expansion
                 
            _bn : _max_coefficientx x number_of_wavelengths numpy array of complex floats
                the array of b coefficients in the Mie expansion
                 
            _cn : _max_coefficientx x number_of_wavelengths numpy array of complex floats
                the array of c coefficients in the Mie expansion
                 
            _dn : _max_coefficientx x number_of_wavelengths numpy array of complex floats
                the array of d coefficients in the Mie expansion
            
                  
             

            Returns
            -------
            None
    
            Examples
             --------
            >>> fill_in_with_actual_example!
        """
        pass
    
    def _compute_s_jn(self, n, z):
        """ Compute the spherical bessel function from the Bessel function
            of the first kind
            
            Arguments
            ---------
            n : int
                order of the bessel function
                
            z : float
                size parameter of the sphere
            
            
            Returns
            -------
            _s_jn
            
        """
        
    def _compute_s_yn(self, n, z):
        """ Compute the spherical bessel function from the Bessel function
            of the first kind
            
            Arguments
            ---------
            n : int
                order of the bessel function
                
            z : float
                size parameter of the sphere
            
            
            Returns
            -------
            _s_jn
            
        """
    def _compute_s_hn(self, n, z):
        """ Compute the spherical bessel function h_n^{(1)}
            
            Arguments
            ---------
            n : int
                order of the bessel function
                
            z : float
                size parameter of the sphere
            
            
            Returns
            -------
            _s_hn
        """

    def _compute_z_jn_prime(self, n,z):
        """ Compute derivative of z*j_n(z) using recurrence relations
        
            Arguments
            ---------
            n : int
                order of the bessel functions
            z : float
                size parameter of the sphere
                
            Returns
            -------
            _z_jn_prime
            
        """ 
    def _compute_z_hn_prime(self, n,z):
        """ Compute derivative of z*h_n^{(1)}(z) using recurrence relations
           
            Arguments
            ---------
            n : int
                order of the bessel functions
            z : float
                size parameter of the sphere
                
            Returns
            -------
            _z_hn_prime
        
        """
        
    def mie_coeff(n, m, mu, x):
        """ COMPLETE THIS DOCSTRING!
            computes the Mie theory a, b, c, and d coefficients for a
            given order n, relative refractive index m, relative permeability mu, and
            size parameter x.  Returns coefficients as an array. 
            
        """
  
        
    
