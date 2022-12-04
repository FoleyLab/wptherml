import numpy as np
from scipy.special import spherical_jn
from scipy.special import spherical_yn
from scipy.special import jv
from scipy.special import yv
from .spectrum_driver import SpectrumDriver
from .materials import Materials


class CuboidDriver(SpectrumDriver, Materials):
    """Compute the absorption, scattering, and extinction spectra of a sphere using Mie theory

    Attributes
    ----------
    Lx : float
        Length of cuboid along x direction

    Ly : float
        Length of the cuboid along y direction
    
    Lz : float
        Length of the cuboid along the z direction

    _a : float
        Lx / 2
    
    _b : float
        Ly / 2

    _c : float
        Lz / 2

    number_of_wavelengths : int
        the number of wavelengths over which the cross sections / efficiencies will be computed

    wavelength_array : 1 x number_of_wavelengths numpy array of floats
        the array of wavelengths in meters over which you will compute the spectra

    _size_factor_array : 1 x number_of_wavelengths numpy array of floats
        size factor of the sphere

    _refractive_index_array : 1 x number_of_wavelengths numpy array of complex floats
        the array of refractive index values corresponding to wavelength_array

    _medium_refractive_index : float
        the refractive index of the surrounding medium - assumed to be real and wavelength-independent

    _epsilon : 1 x number_of_wavelengths numpy array of complex floats
        epsilon of the material = _refractive_index_array ** 2

    _epsilon_B : float
        epsilon of surrounding medium = _medium_refractive_index ** 2

    sigma_scat : numpy array of floats
        the scattering efficiency as a function of wavelength

    sigma_ext : numpy array of floats
        the extenction efficiency as a function of wavelength

    sigma_abs : 1 x number_of_wavelengths numpy array of floats
        the absorption efficiency as a function of wavelength


    Returns
    -------
    None

    Examples
    --------
    >>> fill_in_with_actual_example!
    """

    def __init__(self, args):
        self.parse_input(args)
        self._ci = 0 + 1j

        self.set_refractive_indicex_array()
        self.compute_spectrum()

    def parse_input(self, args):
        if "Lx" in args:
            self.Lx = args["Lx"]
        else:
            self.Lx = 100e-9

        if "Ly" in args:
            self.Ly = args["Ly"]
        else:
            self.Ly = 100e-9
        
        if "Lx" in args:
            self.Lz = args["Lx"]
        else:
            self.Lz = 100e-9

        self._a = self.Lx / 2
        self._b = self.Ly / 2
        self._c = self.Lz / 2

        self._beta = 12.6937 * self._a ** 2
        self.number_of_layers = 3
        

        if "wavelength_list" in args:
            lamlist = args["wavelength_list"]
            self.wavelength_array = np.linspace(lamlist[0], lamlist[1], int(lamlist[2]))
            self.number_of_wavelengths = int(lamlist[2])
            self.wavenumber_array = 1 / self.wavelength_array
        # default wavelength array
        else:
            self.wavelength_array = np.linspace(400e-9, 800e-9, 10)
            self.number_of_wavelengths = 10
            self.wavenumber_array = 1 / self.wavelength_array

        if "material" in args:
            self.material = args["material"]
        else:
            self.material = "ag"

        if "medium_material" in args:
            self.medium_material = args["medium_material"]
        else:
            self.medium_material = "air"

        self._refractive_index_array = np.ones(
            (self.number_of_wavelengths, 3), dtype=complex
        )

        self.q_ext = np.zeros_like(self.wavelength_array)
        self.q_scat = np.zeros_like(self.wavelength_array)
        self.q_abs = np.zeros_like(self.wavelength_array)

    def set_refractive_indicex_array(self):
        """once materials are specified, define the refractive_index_array values"""

        # terminal layers can be air or water now
        _lmed = self.medium_material.lower()

        if _lmed == "air":
            self.material_Air(0)
        elif _lmed == "water":
            self.material_H2O(0)
        elif _lmed == "h2o":
            self.material_H2O(0)

        _lm = self.material.lower()

        # check all possible values of the material string
        # and set material as appropriate.
        # in future probably good to create a single wrapper
        # function in materials.py that will do this so
        # that MieDriver and TmmDriver can just use it rather
        # than duplicating this kind of code in both classes
        if _lm == "air":
            self.material_Air(1)
        elif _lm == "ag":
            self.material_Ag(1)
        elif _lm == "al":
            self.material_Al(1)
        elif _lm == "al2o3":
            self.material_Al2O3(1)
        elif _lm == "aln":
            self.material_AlN(1)
        elif _lm == "au":
            self.material_Au(1)
        elif _lm == "hfo2":
            self.material_HfO2(1)
        elif _lm == "pb":
            self.material_Pb(1)
        elif _lm == "polystyrene":
            self.material_polystyrene(1)
        elif _lm == "pt":
            self.material_Pt(1)
        elif _lm == "re":
            self.material_Re(1)
        elif _lm == "rh":
            self.material_Rh(1)
        elif _lm == "ru":
            self.material_Ru(1)
        elif _lm == "si":
            self.material_Si(1)
        elif _lm == "sio2":
            self.material_SiO2(1)
        elif _lm == "ta2O5":
            self.material_Ta2O5(1)
        elif _lm == "tin":
            self.material_TiN(1)
        elif _lm == "tio2":
            self.material_TiO2(1)
        elif _lm == "w":
            self.material_W(1)
        # default is SiO2
        else:
            self.material_SiO2(1)

    def compute_spectrum(self):
        """Will prepare the attributes forcomputing q_ext, q_abs, q_scat, c_abs, c_ext, c_scat
        via computing the mie coefficients

        Attributes
        ---------
        TBD


        Returns
        -------
        TBD

        """
        for i in range(0, len(self.wavelength_array)):
            # get Mie coefficients... stored in attriubutes
            _n = self._refractive_index_array[i, 1]
            _nB = self._refractive_index_array[i, 0]
            _kb = self.wavenumber_array[i] * _nB 
            self._compute_alpha(_nB ** 2, _n ** 2, _kb)

            # compute q_scat
            self.q_scat[i] = _kb ** 4 / (6 * np.pi) * np.real( np.conj( self._alpha ) * self._alpha  )
            self.q_abs[i] = _kb * np.imag( self._alpha )

            self.q_ext[i] = self.q_scat[i] + self.q_abs[i]

    def _compute_Omega(self):
        """Compute the Eq. (2) in https://doi.org/10.1088/1367-2630/15/6/063013 

        Attributes
        ----------
        self.Omega : float
            the solid angle subtended by the side perpendicular 
            to the polarization axis of the cuboid (the x-axis in this case)


        Test Implemented
        ----------------
        No

        """
        _numerator = self._b * self._c 
        _denominator = np.sqrt( (self._a ** 2 + self._b ** 2) * (self._a ** 2 + self._c ** 2) )

        self._Omega = 4 * np.arcsin( _numerator / _denominator ) 

    def _compute_delta(self, _epsilon_B, _epsilon):
        """ is a term that takes into account the polarization charges 
        at the planar ends of the cuboid orthogonal to the x direction, and is expressed by

        Arguments
        ---------
        _epsilon_B : float
            permittivity of surrounding medium

        _epsilon : complex 
            permittivity of the material at a particular wavelength

        _delta : complex 


        Test Implemented
        ----------------
        No

        """
        _numerator = 8 * self._a * self._b * self._c
        _denominator = (self._a ** 2 + self._b **2 + self._c ) ** (3/2)
        self._delta =  _numerator / _denominator * _epsilon_B / _epsilon

    def _compute_alpha(self, _epsilon_B, _epsilon, _kb):
        """ Polarizability of the cuboid

        Arguments
        ---------
        _epsilon_B : float
            permittivity of surrounding medium

        _epsilon : complex 
            permittivity of the material at a particular wavelength

        _kb : float
            wavenumber in the surrounding medium - n_B * k_0 

        Test Implemented
        ----------------
        No

        """
        _ci = 0+1j
        self._compute_Omega()
        self._compute_delta(_epsilon_B, _epsilon)
        _abc = self._a * self._b * self._c 
        _numerator = 8 * _abc 
        _denominator_1 = _epsilon_B / (_epsilon - _epsilon_B)
        _denominator_2 = -2 * self._Omega - self._delta + _kb ** 2 / 2  * self._beta + 16 / 3 * _ci * _kb ** 3 * _abc
        self._alpha = _numerator / (_denominator_1 - _denominator_2 / (4 * np.pi)) 
