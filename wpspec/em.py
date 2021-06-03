from .interface import Interface
import numpy as np

class Tmm(Interface):
    
    def __init__(self, args):
        super().__init__(args)
        """
        Initilializer for the Tmm class which inherits from Interface

        Will solve transfer matrix equations for the structure defined

        Parameters
        ----------
        args : dictionary containing keywords and values that determine what structure we are modeling
               and what we want to calculate

        Returns
        -------
        None.

        """
        
    def transfer_matrix(self):
        self.reflectivity_array = np.zeros_like(self.wavelength_array)
        print(self.reflectivity_array)
        return 1

