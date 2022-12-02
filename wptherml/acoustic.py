import numpy as np
from scipy.special import spherical_jn
from scipy.special import spherical_yn
from scipy.special import jv
from scipy.special import yv
from .spectrum_driver import SpectrumDriver
from .materials import Materials


class AcousticDriver(SpectrumDriver, Materials):
    """Compute the absorption, scattering, and extinction spectra of a sphere using Mie theory

    Attributes
    ----------
    Lx : float
        length of slab along x

    Ly : float
        length of slab along y

    k_array : numpy array 
        the array wavenumber values

    k_x : float
        the x-component of wavenumber

    k_y : float
        the y-component of wavenumber

    k_z : flaot
        the z-component of wavenumber 

    gamma : float
        the coincidence value

    m : integer
        mode number along x

    n : integer
        mode number along y

    S_mn : float
        radiation efficiency for mode (m, n) at 



    Returns
    -------
    None

    Examples
    --------
    >>> fill_in_with_actual_example!
    """

    def __init__(self, args):
        self.parse_input(args)
        print("Lx is ", self.Lx)

        # hard-code theta array
        self.theta = np.linspace(0, np.pi/2, 200)
        # hard-code phi array
        self.phi = np.linspace(0, np.pi * 2, 200)

        #self.ci = 0 + 1j
        #self.compute_spectrum()

    def parse_input(self, args):
        if "Lx" in args:
            self.Lx = args["Lx"]
        else:
            self.Lx = 100 # ==> To-do: decide what a reasonable value is!
        
        if "Ly" in args:
            self.Ly = args["Ly"]
        else:
            self.Ly = 100 # ==> To-do: decide what a reasonable value is!

        self.k = 100 # ==> To-do: decide what reasonable values are!

    def _compute_gamma(self, m, n):
        """Compute the spherical bessel function from the Bessel function
        of the first kind

        Arguments
        ---------
        m : integer
            the mode number along x

        n : integer
            the mode number along y

        gamma : float (to be computed)
            the coincidence number 
        Notes
        -----
        Equation for gamma can be found on 2.180 on page 69 of Earl Williams Book.

        - the value of k is stored in the attribute self.k
        - the value of Lx is stored in the attribute self.Lx
        - the value of Ly is stored in the attribute self.Ly

        """
<<<<<<< Updated upstream
=======
        self.gamma = self.k / (np.sqrt((m*np.pi/self.Lx)**2 + (n*np.pi/self.Ly)**2))

>>>>>>> Stashed changes
        pass # <=== replace with lines of code to compute self.gamma

    def _compute_kx_ky_kz(self):
        """Compute the kx, ky, and kz values given

        Arguments
        ---------
        theta : numpy array of floats
            theta value in radians

        phi : numpy array of floats
            phi value in radians

        k_x : 2D numpy array of floats (to be computed)
            x-components of wavenumber 

        k_y : 2D numpy array of floats (to be computed)
            y-component of wavenumber 

        k_z : 2D numpy array of floats (to be computed)
            z-component of wavenumber 

        Notes
        -----

        Equations for kx, ky, kz in terms of k, theta, and phi can be found in Eq. 2.47 
        on page 29 of Earl Williams book.

        - the value of k is stored in the attribute self.k
        - the values of theta are stored in the attribute self.theta
        - the values of phi are stored in the attribute self.phi

        self.k_x, self.k_y, and self.k_z should be 2D numpy arrays with dimensions len(theta) x len(phi)

        """
<<<<<<< Updated upstream
        pass # <=== replace with lines of code to compute self.k_x, self.k_y, and self.k_z 
=======
    
        self.k_x_array = self.k * np.cos(self.phi)*np.sin(self.theta)
        self.k_y_array = self.k * np.sin(self.phi)*np.sin(self.theta)
        self.k_z_array = self.k * np.cos(self.theta)



       # <=== replace with lines of code to compute self.k_x, self.k_y, and self.k_z 
>>>>>>> Stashed changes


    def compute_spectrum(self):
        """Will prepare the attributes and compute S_mn

        Attributes
        ---------
        TBD


        Returns
        -------
        TBD

        Notes
        -----
        Equation for S_mn can be found on Eq. 2.179 on page 69 of Earl Williams book. 

        """
        pass 


