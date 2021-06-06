from .spectrum_driver import SpectrumDriver
import numpy as np



class TmmDriver(SpectrumDriver):
    """ Compute the absorption, scattering, and extinction spectra of a sphere using Mie theory
    
        Attributes
        ----------
            number_of_layers : int
                the number of layers in the multilayer
            
            number_of_wavelengths : int
                the number of wavelengths in the wavelength_array
                
            thickness_array : 1 x number_of_layers numpy array of floats
                the thickness of each layer
                
            material_array : 1 x number_of_layers numpy array of str
                the materia of each layer
                
             wavelength_array : numpy array of floats
                 the array of wavelengths in meters over which you will compute the spectra

             reflectivity_array : 1 x number_of_wavelengths numpy array of floats
                  the reflection spectrum 

             transmissivity_array : 1 x number_of_wavelengths numpy array of floats
                  the transmission spectrum

             emissivity_array : 1 x number_of_wavelengths numpy array of floats
                  the absorptivity / emissivity spectrum
                  
              _refractive_index_array : number_of_layers x number_of_wavelengths numpy array of complex floats
                 the array of refractive index values corresponding to wavelength_array
                 
              _tm : 2 x 2 x number_of_wavelengths numpy array of complex floats
                 the transfer matrix for each wavelength
                 
              _pm : 2 x 2 x (number_of_layers-2) x number_of_wavelengths numpy array of complex floats
                 the P matrix for each of the finite-thickness layers for each wavelength
                 
              _dm : 2 x 2 x number_of_layers x number_of_wavelengths numpy array of complex floats
                 the D matrix for each of the layers for each wavelength
                 
              _dim : 2 x 2 x number_of_layers x number_of_wavelengts numpy array of complex floats
                 the inverse of the D matrix for each of the layers for each wavelength

        Returns
        -------
            None
    
    """
    def __init__(self, thickness):
        """ constructor for the TmmDriver class
    
            Assign values for attributes thickness_array, material_array then call
            compute_spectrum to compute values for attributes reflectivity_array, 
            transmissivity_array, and emissivity_array
        
        """
        
        
        ''' hard-coded a lot of this for now, we will obviously generalize soon! '''
        self.thickness = thickness
        self.number_of_wavelengths = 100
        self.number_of_layers = 3
        self.wavelength_array = np.linspace(400e-9, 800e-9, self.number_of_wavelengths)
        self.thickness_array = np.array([0, thickness, 0])
        self.polarization = 's'
        self.theta_array = np.array([0, 0, 0])
        
        # pretty pythonic way to create the _refractive_index_array
        # that will result in self._refractive_index_array[1, 3] -> RI of layer index 1 (2nd layer) 
        # at wavelength index 3 (4th wavelength in wavelength_array)
        self._refractive_index_array = np.reshape(np.tile(np.array([1+0j, 1.5+0j, 1+0j]), self.number_of_wavelengths), 
                                                  (self.number_of_wavelengths, self.number_of_layers)) 
        


    def compute_spectrum(self):
        """ computes the following attributes:
            
            Attributes
            ----------
                reflectivity_array 
            
                transmissivity_array
            
                emissivity_array
            
            Returns
            -------
                None
                
            
            Will compute attributes by 
            
                - calling _compute_tm method
                
                - evaluating r amplitudes from _tm
                
                - evaluationg R from rr*
                
                - evaluating t amplitudes from _tm
                
                - evaluating T from tt* n_L cos(\theta_L) / n_1 cos(\theta_L)
            
        
        
        """      
        
    
    def _compute_tm(self):
        return 1
    
    def _compute_pm(self):
        return 1
    
    def _compute_dm(self):
        return 1

