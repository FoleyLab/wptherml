import numpy as np
from scipy.special import spherical_jn
from scipy.special import spherical_yn
from scipy.special import jv
from scipy.special import yv
from .spectrum_driver import SpectrumDriver
from .materials import Materials


class MieDriver(SpectrumDriver, Materials):
    """Compute the absorption, scattering, and extinction spectra of a sphere using Mie theory

    Attributes
    ----------
    radius : float
        the radius of the sphere

    number_of_wavelengths : int
        the number of wavelengths over which the cross sections / efficiencies will be computed

    wavelength_array : 1 x number_of_wavelengths numpy array of floats
        the array of wavelengths in meters over which you will compute the spectra

    _size_factor_array : 1 x number_of_wavelengths numpy array of floats
        size factor of the sphere

    _relative_refractive_index_array : 1 x number_of_wavelengths numpy array of complex floats
        the array of refractive index values corresponding to wavelength_array

    _medium_refractive_index : float
        the refractive index of the surrounding medium - assumed to be real and wavelength-independent

    q_scat : numpy array of floats
        the scattering efficiency as a function of wavelength

    q_ext : numpy array of floats
        the extenction efficiency as a function of wavelength

    q_abs : 1 x number_of_wavelengths numpy array of floats
        the absorption efficiency as a function of wavelength

    c_scat : numpy array of floats
        the scattering cross section as a function of wavelength

    c_ext : numpy array of floats
        the extinction cross section as a function of wavelength

    c_abs : 1 x number_of_wavelengths numpy array of floats
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

    def __init__(self, args):
        self.parse_input(args)
        print("Radius of the sphere is ", self.radius)
        self.ci = 0 + 1j

        self.set_refractive_indicex_array()
        self.compute_spectrum()

    def parse_input(self, args):
        if "radius" in args:
            self.radius = args["radius"]
        else:
            self.radius = 100e-9

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

        if "sphere_material" in args:
            self.sphere_material = args["sphere_material"]
        else:
            self.sphere_material = "ag"

        if "medium_material" in args:
            self.medium_material = args["medium_material"]
        else:
            self.medium_material = "air"

        self.number_of_layers = 3
        self._refractive_index_array = np.ones(
            (self.number_of_wavelengths, 3), dtype=complex
        )
        self._relative_permeability = 1.0 + 0j
        self._size_factor_array = np.pi * 2 * self.radius / self.wavelength_array

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

        _lm = self.sphere_material.lower()

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

        self._relative_refractive_index_array = (
            self._refractive_index_array[:, 1] / self._refractive_index_array[:, 0]
        )

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
            m_val = self._relative_refractive_index_array[i]
            mu_val = self._relative_permeability
            x_val = self._size_factor_array[i]
            self._compute_mie_coeffients(m_val, mu_val, x_val)

            # compute q_scat
            self.q_scat[i] = self._compute_q_scattering(x_val)
            self.q_ext[i] = self._compute_q_extinction(x_val)
            self.q_abs[i] = self.q_ext[i] - self.q_scat[i]

    def _compute_s_jn(self, n, z):
        """Compute the spherical bessel function from the Bessel function
        of the first kind

        Arguments
        ---------
        n : 1 x _max_coefficient numpy array of ints
            orders of the bessel function

        z : float
            size parameter of the sphere


        Returns
        -------
        _s_jn

        Test Implemented
        ----------------
        Yes

        """
        ns = n + 0.5
        return np.sqrt(np.pi / (2 * z)) * jv(ns, z)

    def _compute_s_yn(self, n, z):
        """Compute the spherical bessel function from the Bessel function
        of the first kind

        Arguments
        ---------
        n : 1 x _max_coefficient numpy array of ints
            orders of the bessel function

        z : float
            variable passed to the bessel function


        Returns
        -------
        _s_jn

        Test Implemented
        ----------------
        Yes

        """
        ns = n + 0.5
        return np.sqrt(np.pi / (2 * z)) * yv(ns, z)

    def _compute_s_hn(self, n, z):
        """Compute the spherical bessel function h_n^{(1)}

        Arguments
        ---------
        n : 1 x _max_coefficient array of int
            orders of the bessel function

        z : float
            variable passed to the bessel function


        Returns
        -------
        _s_hn

        Test Implemented
        ----------------
        Yes
        """
        return spherical_jn(n, z) + self.ci * spherical_yn(n, z)

    def _compute_z_jn_prime(self, n, z):
        """Compute derivative of z*j_n(z) using recurrence relations

        Arguments
        ---------
        n : 1 x _max_coefficient array of int
            orders of the bessel functions
        z : float
            variable passed to the bessel function

        Returns
        -------
        _z_jn_prime

        Test Implemented
        ----------------
        Yes

        """
        return z * spherical_jn(n - 1, z) - n * spherical_jn(n, z)

    def _compute_z_hn_prime(self, n, z):
        """Compute derivative of z*h_n^{(1)}(z) using recurrence relations

        Arguments
        ---------
        n : 1 x _max_coefficient array of int
            orders of the bessel functions
        z : float
            variable passed to the bessel function

        Returns
        -------
        _z_hn_prime

        """

        return z * self._compute_s_hn(n - 1, z) - n * self._compute_s_hn(n, z)

    def _compute_mie_coeffients(self, m, mu, x):
        """computes the Mie coefficients given relative refractive index,

        Arguments
        ---------
        n : 1 x _max_coefficient array of ints
            order of the mie coefficients functions
        m : complex float
            relative refractive index of the sphere to the medium
        mu : complex float
            relative permeability of the sphere to the medium (typically 1)
        x : float
            size parameter of the sphere

        Attributes
        -------
        _an

        _bn

        _cn

        _dn

        """
        self._compute_n_array(x)
        # self._n_array will be an array from 1 to n_max

        # pre-compute terms that will be used numerous times in computing coefficients
        _jnx = spherical_jn(self._n_array, x)
        _jnmx = spherical_jn(self._n_array, m * x)
        _hnx = self._compute_s_hn(self._n_array, x)
        _xjnxp = self._compute_z_jn_prime(self._n_array, x)
        _mxjnmxp = self._compute_z_jn_prime(self._n_array, m * x)
        _xhnxp = self._compute_z_hn_prime(self._n_array, x)

        # a_n coefficients
        _a_numerator = m**2 * _jnmx * _xjnxp - mu * _jnx * _mxjnmxp
        _a_denominator = m**2 * _jnmx * _xhnxp - mu * _hnx * _mxjnmxp

        self._an = _a_numerator / _a_denominator

        # b_n coefficients
        _b_numerator = mu * _jnmx * _xjnxp - _jnx * _mxjnmxp
        _b_denominator = mu * _jnmx * _xhnxp - _hnx * _mxjnmxp

        self._bn = _b_numerator / _b_denominator

        # c_n coefficients
        _c_numerator = mu * _jnx * _xhnxp - mu * _hnx * _xjnxp
        _c_denominator = mu * _jnmx * _xhnxp - _hnx * _mxjnmxp

        self._cn = _c_numerator / _c_denominator

        # d_n coefficients
        _d_numerator = mu * m * _jnx * _xhnxp - mu * m * _hnx * _xjnxp
        _d_denominator = m**2 * _jnmx * _xhnxp - mu * _hnx * _mxjnmxp

        self._dn = _d_numerator / _d_denominator
        # return [self._an,self._bn,self._cn,self._dn]

    def _compute_q_scattering(self, x):
        """computes the scattering efficiency from the mie coefficients

        Parameters
        ----------
        m : complex float
            relative refractive index of the sphere

        mu : float
            relative permeability of the sphere

        x : float
            size parameter of the sphere, defined as 2 * pi * r / lambda
            where r is the radius of the sphere and lambda is the wavelength of illumination

        Attributes
        -------
        q_scat

        Returns
        -------
        q_scat

        """
        # c = self._compute_mie_coeffients(m, mu, x)
        an = self._an
        bn = self._bn

        q_scat = 0.0
        for i in range(0, len(an)):
            # n is i + 1
            # because i indexes the arrays, n is the multipole order
            n = i + 1
            q_scat = q_scat + 2 / x**2 * (2 * n + 1) * (
                np.abs(an[i]) ** 2 + np.abs(bn[i]) ** 2
            )

        return q_scat

    def _compute_q_extinction(self, x):
        """computes the extinction efficiency from the mie coefficients

        Parameters
        ----------

        m : complex float
            relative refractive index of the sphere

        mu : float
            relative permeability of the sphere

        x : float
            size parameter of the sphere, defined as 2 * pi * r / lambda
            where r is the radius of the sphere and lambda is the wavelength of illumination

        Attributes
        -------
        q_ext

        Returns
        -------
        q_ext

        """

        # c = self._compute_mie_coeffients(m, mu, x)
        an = self._an
        bn = self._bn

        q_ext = 0
        for i in range(0, len(an)):
            # n is i + 1
            # because i indexes the arrays, n is the multipole order
            n = i + 1
            q_ext = q_ext + 2 / x**2 * (2 * n + 1) * np.real(an[i] + bn[i])

        return q_ext

    def _compute_n_array(self, x):
        _n_max = int(x + 4 * x ** (1 / 3.0) + 2)
        self._n_array = np.copy(np.linspace(1, _n_max, _n_max, dtype=int))

    def _compute_gl(l, r, h, mu, omega_p, eps_inf, eps_d):
        """docstring goes here!"""
        zeta = h / r

        omega_l = self._compute_omega_l(l, omega_p, eps_inf, eps_d)
        # note in atomics, 4 * pi * epsilon_0 = 1
        g_l_squared = mu * omega_p / (1 * h * r**3) * (omega_l / omega_p) ** 3
        g_l_squared *= (l + 1) ** 2 / ((1 + zeta) ** (2 * l + 4)) * (1 + 1 / (2 * l))

        return np.sqrt(g_l_squared), omega_l

    def _compute_gm(l_max, r, h, mu, omega_p, eps_inf, eps_d):
        gm_array = np.zeros(len(l_max) - 2)
        omegam_array = np.zeros(len(l_max) - 2)
        for i in range(2, l_max):
            gm_array[i - 2], omegam_array[i - 2] = self._compute_gl(
                i, r, h, mu, omega_p, eps_inf, eps_d
            )

        gm = np.sqrt(np.sum(gm_array * gm_array))
        omegam = np.sum(omegam_array * gm_array * gm_array) / np.sum(
            gm_array * gm_array
        )

        return gm, omegam

    def _compute_omega_l(l, omega_p, eps_inf, eps_d):
        """docstring goes here!"""
        return omega_p / np.sqrt(eps_inf + (1 + 1 / l) * eps_d)

    def compute_hamiltonian(self, N, h, drude_dictionary, emitter_dictionary):
        """docstring goes here!"""
        # this is the wavelength relevant for the system studied in PRL 112, 253601 (2014)
        # it is not general! should think of a more general way to get the l_max value!
        # in that paper, the set omega_0 to 2.5 eV so we can convert that frequency into a wavelength
        lambda_0 = 1240 / 2.5
        # now we can get the size parameter
        x = 2 * np.pi * self.radius / lambda_0
        _l_max = int(x + 4 * x ** (1 / 3.0) + 2)

        # get parameters of the metal nanoparticle
        _omega_p = drude_dictionary["omega_p"]
        _eps_inf = drude_dictionary["eps_inf"]
        _gamma_p = drude_dictionary["gamma_p"]
        _eps_d = drude_dictionary["eps_d"]

        # get parameters of the quantum emitter
        _gamma_qe = emitter_dictionary["gamma_qe"]
        _omega_0 = emitter_dictionary["omega_0"]
        _mu = emitter_dictionary["mu"]

        # compute g_1 and omega_1
        _g1, _omega1 = self._compute_gl(
            1, self.radius, h, _mu, _omega_p, _eps_inf, _eps_d
        )

        # compute g_M and omega_M
        _gm, _omegam = self._compute_gm(
            _l_max, self.radius, h, _mu, _omega_p, _eps_inf, _eps_d
        )

        H = np.zeros((3, 3), dtype=complex)
        ci = 0 + 1j

        # diagonal elements
        H[0, 0] = _omega_0 - ci * _gamma_qe / 2
        H[1, 1] = _omega1 - ci * _gamma_p / 2
        H[2, 2] = _omegam - ci * _gamma_p / 2

        # off-diagonal for coupling to dipolar term
        H[0, 1] = _g1 * np.sqrt(N / 3)
        H[1, 0] = _g1 * np.sqrt(N / 3)

        # off-diagonal for coupling to all higher multipoles
        H[0, 2] = _gm
        H[2, 0] = _gm
