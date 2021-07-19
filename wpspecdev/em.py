from .spectrum_driver import SpectrumDriver
from .materials import Materials
import numpy as np


class TmmDriver(SpectrumDriver, Materials):
    """Compute the absorption, scattering, and extinction spectra of a sphere using Mie theory

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

        incident_angle : float
            the incident angle of light relative to the normal to the multilayer (0 = normal incidence!)

        polarization : str
            indicates if incident light is 's' or 'p' polarized

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

        _kz_array : 1 x number_lf_layers x number_of_wavelengths numpy array of complex floats
            the z-component of the wavevector in each layer of the multilayer for each wavelength

        _k0_array : 1 x number_of_wavelengths numpy array of floats
            the wavevector magnitude in the incident layer for each wavelength

        _kx_array : 1 x number_of_wavelengths numpy array of floats
            the x-component of the wavevector for each wavelength (conserved throughout layers)

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

    def __init__(self, args):
        """constructor for the TmmDriver class

        Assign values for attributes thickness_array, material_array then call
        compute_spectrum to compute values for attributes reflectivity_array,
        transmissivity_array, and emissivity_array

        """
        # parse user inputs
        self.parse_input(args)
        # set refractive index array
        self.set_refractive_indicex_array()
        # compute reflectivity spectrum
        self.compute_spectrum()
        # print output message
        print(
            " Your spectra have been computed! \N{smiling face with sunglasses} "
        )

    def parse_input(self, args):
        """method to parse the user inputs and define structures / simulation

        Returns
        -------
        None

        """
        if "incident_angle" in args:
            # user input expected in deg so convert to radians
            self.incident_angle = args["incident_angle"] * np.pi / 180.0
        else:
            self.incident_angle = 0.0

        if "polarization" in args:
            self.polarization = args["polarization"]
            self.polarization = self.polarization.lower()
        else:
            self.polarization = "p"

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

        # need to throw some exceptions if len(self.thickness_array)!=len(self.material_array)
        if "thickness_list" in args:
            self.thickness_array = args["thickness_list"]
        # default structure
        else:
            print("  Thickness array not specified!")
            print("  Proceeding with default structure - optically thick W! ")
            self.thickness_array = [0, 900e-9, 0]

        if "material_list" in args:
            self.material_array = args["material_list"]
            self.number_of_layers = len(self.material_array)
        else:
            print("  Material array not specified!")
            print("  Proceeding with default structure - Air / SiO2 / Air ")
            self.material_array = ["Air", "SiO2", "Air"]
            self.number_of_layers = 3

    def set_refractive_indicex_array(self):
        """once materials are specified, define the refractive_index_array values"""

        # initialize _ri_list based on the number of layers
        _ri_list = np.ones(self.number_of_layers, dtype=complex)

        # initialize the _refractive_index_array with dummy values define the true values later!
        self._refractive_index_array = np.reshape(
            np.tile(_ri_list, self.number_of_wavelengths),
            (self.number_of_wavelengths, self.number_of_layers),
        )

        # terminal layers default to air for now... generalize later!
        self.material_Air(0)
        self.material_Air(self.number_of_layers - 1)
        for i in range(1, self.number_of_layers - 1):
            # get lower clase version of the material string
            # to avoid any conflicts with variation in cases
            # given by the user
            _lm = self.material_array[i].lower()

            # check all possible values of the material string
            # and set material as appropriate.
            # in future probably good to create a single wrapper
            # function in materials.py that will do this so
            # that MieDriver and TmmDriver can just use it rather
            # than duplicating this kind of code in both classes
            if _lm == "air":
                self.material_Air(i)
            elif _lm == "ag":
                self.material_Ag(i)
            elif _lm == "al":
                self.material_Al(i)
            elif _lm == "al2o3":
                self.material_Al2O3(i)
            elif _lm == "aln":
                self.material_AlN(i)
            elif _lm == "au":
                self.material_Au(i)
            elif _lm == "hfo2":
                self.material_HfO2(i)
            elif _lm == "pb":
                self.material_Pb(i)
            elif _lm == "polystyrene":
                self.material_polystyrene(i)
            elif _lm == "pt":
                self.material_Pt(i)
            elif _lm == "re":
                self.material_Re(i)
            elif _lm == "rh":
                self.material_Rh(i)
            elif _lm == "ru":
                self.material_Ru(i)
            elif _lm == "si":
                self.material_Si(i)
            elif _lm == "sio2":
                self.material_SiO2(i)
            elif _lm == "ta2O5":
                self.material_Ta2O5(i)
            elif _lm == "tin":
                self.material_TiN(i)
            elif _lm == "tio2":
                self.material_TiO2(i)
            elif _lm == "w":
                self.material_W(i)
            # default is SiO2
            else:
                self.material_SiO2(i)

    def compute_spectrum(self):
        """computes the following attributes:

        Attributes
        ----------
            reflectivity_array

            transmissivity_array

            emissivity_array

        Returns
        -------
            None
        """

        # with all of these formed, you can now call _compute_tm()
        self._compute_k0()
        self._compute_kx()
        self._compute_kz()

        # compute the reflectivity in a loop for now!
        self.reflectivity_array = np.zeros_like(self.wavelength_array)
        self.transmissivity_array = np.zeros_like(self.wavelength_array)
        self.emissivity_array = np.zeros_like(self.wavelength_array)

        for i in range(0, self.number_of_wavelengths):
            _k0 = self._k0_array[i]
            _ri = self._refractive_index_array[i, :]
            _kz = self._kz_array[i, :]

            # get transfer matrix, theta_array, and co_theta_array for current k0 value
            _tm, _theta_array, _cos_theta_array = self._compute_tm(
                _ri, _k0, _kz, self.thickness_array
            )

            # reflection amplitude
            _r = _tm[1, 0] / _tm[0, 0]

            # transmission amplitude
            _t = 1 / _tm[0, 0]

            # refraction angle and RI prefractor for computing transmission
            _factor = (
                _ri[self.number_of_layers - 1]
                * _cos_theta_array[self.number_of_layers - 1]
                / (_ri[0] * _cos_theta_array[0])
            )

            # reflectivity
            self.reflectivity_array[i] = np.real(_r * np.conj(_r))

            # transmissivity
            self.transmissivity_array[i] = np.real(_t * np.conj(_t) * _factor)

            # emissivity
            self.emissivity_array[i] = (
                1 - self.reflectivity_array[i] - self.transmissivity_array[i]
            )

    def _compute_kz(self):
        """computes the z-component of the wavevector in each layer of the stack

        Attributes
        ----------
            _refractive_index_array : number_of_layers x number_of_wavelengths numpy array of complex floats
                the array of refractive index values corresponding to wavelength_array

            _kz_array : number_of_layers x number_of_wavelengths numpy array of complex floats
                the z-component of the wavevector in each layer of the multilayer for each wavelength

            _kx_array : 1 x number_of_wavelengths numpy array of complex floats
                the x-component of the wavevector in each layer for each wavelength

            _k0_array : 1 x number_of_wavelengths numpy array of floats
                the wavevector magnitude in the incident layer for each wavelength

        """
        self._kz_array = np.sqrt(
            (self._refractive_index_array * self._k0_array[:, np.newaxis]) ** 2
            - self._kx_array[:, np.newaxis] ** 2
        )

    def _compute_k0(self):
        """computes the _k0_array

        Attributes
        ----------
            wavelength_array : 1 x number of wavelengths float
                the wavelengths that will illuminate the structure in SI units

            _k0_array : 1 x number of wavelengths float
                the wavenumbers that will illuminate the structure in SI units

        """
        self._k0_array = np.pi * 2 / self.wavelength_array

    def _compute_kx(self):
        """computes the _kx_array

        Attributes
        ----------
            _refractive_index_array : number_of_layers x number_of_wavelengths numpy array of complex floats
                the array of refractive index values corresponding to wavelength_array

            incident_angle : float
                the angle of incidence of light illuminating the structure

            _kx_array : 1 x number_of_wavelengths numpy array of complex floats
                the x-component of the wavevector in each layer for each wavelength

            _k0_array : 1 x number_of_wavelengths numpy array of floats
                the wavevector magnitude in the incident layer for each wavelengthhe wavenumbers that will illuminate the structure in SI units
        """
        # compute kx_array
        self._kx_array = (
            self._refractive_index_array[:, 0]
            * np.sin(self.incident_angle)
            * self._k0_array
        )

    def _compute_tm(self, _refractive_index, _k0, _kz, _d):
        """compute the transfer matrix for each wavelength

        Returns
        -------
        _tm : 2 x 2 complex numpy array
            transfer matrix for the _k0 value

        _THETA : 1 x number_of_layers complex numpy array
            refraction angles in each layer for the _k0 value

        _CTHETA : 1 x number_of_layers complex numpy array
            cosine of the refraction angles in each layer for the _k0 value

        """
        _DM = np.zeros((2, 2, self.number_of_layers), dtype=complex)
        _DIM = np.zeros((2, 2, self.number_of_layers), dtype=complex)
        _PM = np.zeros((2, 2, self.number_of_layers), dtype=complex)
        _CTHETA = np.zeros(self.number_of_layers, dtype=complex)
        _THETA = np.zeros(self.number_of_layers, dtype=complex)

        _PHIL = _kz * _d
        _THETA[0] = self.incident_angle
        _CTHETA[0] = np.cos(self.incident_angle)

        _CTHETA[1 : self.number_of_layers] = _kz[1 : self.number_of_layers] / (
            _refractive_index[1 : self.number_of_layers] * _k0
        )
        _THETA[1 : self.number_of_layers] = np.arccos(
            _CTHETA[1 : self.number_of_layers]
        )

        _DM[:, :, 0], _tm = self._compute_dm(_refractive_index[0], _CTHETA[0])

        for i in range(1, self.number_of_layers - 1):
            _DM[:, :, i], _DIM[:, :, i] = self._compute_dm(
                _refractive_index[i], _CTHETA[i]
            )
            _PM[:, :, i] = self._compute_pm(_PHIL[i])
            _tm = np.matmul(_tm, _DM[:, :, i])
            _tm = np.matmul(_tm, _PM[:, :, i])
            _tm = np.matmul(_tm, _DIM[:, :, i])

        (
            _DM[:, :, self.number_of_layers - 1],
            _DIM[:, :, self.number_of_layers - 1],
        ) = self._compute_dm(
            _refractive_index[self.number_of_layers - 1],
            _CTHETA[self.number_of_layers - 1],
        )

        _tm = np.matmul(_tm, _DM[:, :, self.number_of_layers - 1])

        return _tm, _THETA, _CTHETA

    def _compute_dm(self, refractive_index, cosine_theta):

        """compute the D and D_inv matrices for each layer and wavelength

        Arguments
        ---------
            refractive_index : complex float
                refractive index of the layer you are computing _dm and _dim for

            cosine_theta : complex float
                cosine of the complex refraction angle within the layer you are computing _dm and _dim for

        Attributes
        ----------
            polarization : str
                string indicating the polarization convention of the incident light

        Returns
        -------
        _dm, _dim
        """

        _dm = np.zeros((2, 2), dtype=complex)
        _dim = np.zeros((2, 2), dtype=complex)

        if self.polarization == "s":
            _dm[0, 0] = 1 + 0j
            _dm[0, 1] = 1 + 0j
            _dm[1, 0] = refractive_index * cosine_theta
            _dm[1, 1] = -1 * refractive_index * cosine_theta

        elif self.polarization == "p":
            _dm[0, 0] = cosine_theta + 0j
            _dm[0, 1] = cosine_theta + 0j
            _dm[1, 0] = refractive_index
            _dm[1, 1] = -1 * refractive_index

        # Note it is actually faster to invert the 2x2 matrix
        # "By Hand" than it is to use linalg.inv
        # and this inv step seems to be the bottleneck for the TMM function
        # but numpy way would just look like this:
        # _dim = inv(_dm)
        _tmp = _dm[0, 0] * _dm[1, 1] - _dm[0, 1] * _dm[1, 0]
        _det = 1 / _tmp
        _dim[0, 0] = _det * _dm[1, 1]
        _dim[0, 1] = -1 * _det * _dm[0, 1]
        _dim[1, 0] = -1 * _det * _dm[1, 0]
        _dim[1, 1] = _det * _dm[0, 0]

        return _dm, _dim

    def _compute_pm(self, phil):
        """compute the P matrices for each intermediate-layer layer and wavelength

        Arguments
        ---------
            phil : complex float
                kz * d of the current layer

        Returns
        -------
        _pm
        """

        _pm = np.zeros((2, 2), dtype=complex)
        _ci = 0 + 1j
        _a = -1 * _ci * phil
        _b = _ci * phil

        _pm[0, 0] = np.exp(_a)
        _pm[1, 1] = np.exp(_b)

        return _pm
