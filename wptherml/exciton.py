import numpy as np
from .spectrum_driver import SpectrumDriver


class ExcitonDriver(SpectrumDriver):
    """ A class for computing the dynamics and spectra of a system modelled by the Frenkel Exciton Hamiltonian

    Attributes
    ----------
    radius : float
        the radius of the sphere


    Returns
    -------
    None

    Examples
    --------
    >>> fill_in_with_actual_example!
    """

    def __init__(self, args):
        self.parse_input(args)
        print("Exciton Energy is  ", self.exciton_energy)


    def parse_input(self, args):
        if "exciton_energy" in args:
            self.exciton_energy = args["exciton_energy"] 
        else:
            self.exciton_energy = 0.5
        if "number_of_monomers" in args:
            self.number_of_monomers = args["number_of_monomers"]
        else:
            self.number_of_monomers = 2
        if "displacement_between_monomers" in args:
            self.displacement_between_monomers = args['displacement_between_monomers']
        else:
            self.displacement_between_monomers = np.array([1, 0, 0])

        if "transition_dipole_moment" in args:
            self.transition_dipole_moment = args["transition_dipole_moment"]
        else:
            self.transition_dipole_moment = np.array([0, 0, 1])
        if "refractive_index" in args:
                self.refractive_index = args["refractive_index"]
        else:
            self.refractive_index = 1

        self.coords = np.zeros((3, self.number_of_monomers))

        for i in range(self.number_of_monomers):
            self.coords[:,i] = self.displacement_between_monomers * i


    
    def _compute_H0_element(self, n, m):
        return self.exciton_energy * (n == m)

    def _compute_dipole_dipole_coupling(self, n, m):
        """ Method to compute the dipole-dipole potential between excitons located on site n and site m
        
        Arguments
        ---------
        n : int
            the index of site n offset by +1 relative to the python index
        m : int
            the index of site m offset by +1 relative to the python index

        Attributes
        ----------
        coords : 3 x number_of_monomers numpy array of floats
            the cartesian coordinates of each monomer

        transition_dipole_moment : 1x3 numpy array of floats
            the transition dipole moment associated with the excitons

        Returns
        -------
        V_nm : float
             the dipole-dipole potential between exciton on site n and m
        """ 
        # offset the indices for python
        _n = n - 1
        _m = m - 1

        # calculate separation vector between site m and site n
        _r_vec = self.coords[:, _m] - self.coords[:, _n]

        # self.transition_dipole_moment is the transition dipole moment!
        V_nm = (1 / (self.refractive_index ** 2 * np.sqrt(np.dot(_r_vec, _r_vec)) ** 3 )) * (np.dot(self.transition_dipole_moment, self.transition_dipole_moment) - 3 * ((np.dot(self.transition_dipole_moment, _r_vec) * np.dot(_r_vec, self.transition_dipole_moment)) / (np.sqrt(np.dot(_r_vec, _r_vec)) ** 2))) 

        return V_nm
        

    def compute_spectrum(self):
        """Will prepare the Frenkel Exciton Hamiltonian and use to compute an absorption spectrum 

        Attributes
        ---------
        TBD


        Returns
        -------
        TBD

        """
        pass

