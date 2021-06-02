"""
interface.py
A python package for modeling light-matter interactions!

Handles the primary functions
"""

import numpy as np
class interface:
    

    def __init__(self, args):
        """
        Initilializer for the interface class

        Will decide what structure we are dealing with and what we want to compute!

        Parameters
        ----------
        args : dictionary containing keywords and values that determine what structure we are modeling
               and what we want to calculate

        Returns
        -------
        None.

        Default values of attributes for first pass:
        self.number_of_wavelengths : 400
        self.wavelength_array[:]  : np.linspace(400e-9,800e-9,self.number_of_wavelengths) (array of wavelengths between 400 and 800 nm) 
        self.number_of_layers : 3
        self.refractive_index_array[:] : np.array([1+0j, 1.5 + 0j, 1+0j]) (air, glass, air)
        self.thickness_array[:] : np.array([0, 100e-9, 0]) (infinite, 100 nm, infinite)
        """
        self.number_of_wavelengths = 400
        self.wavelength_array = np.linspace(400e-9, 800e-9, self.number_of_wavelengths)
        self.number_of_layers = 3
        self.refractive_index_array = np.array([1.0+0j, 1.5+0j, 1.0+0j])
        self.thickness_array = np.array([0, 100e-9, 0])
    

    def canvas(self):
        """
        will print values of attributes!
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
        print("Number of wavelengths", self.number_of_wavelengths)
        print("number of layers",self.number_of_layers)
        

#if __name__ == "__main__":
#    # Do something if this file is invoked on its own

test_ml = interface([])
print(test_ml.canvas())
