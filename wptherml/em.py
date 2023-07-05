from .spectrum_driver import SpectrumDriver
from .materials import Materials
from .therml import Therml
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import matplotlib.colors as colors
import matplotlib.cm as cmx


class TmmDriver(SpectrumDriver, Materials, Therml):
    """Collects methods for computing the reflectivity, absorptivity/emissivity, and transmissivity
       of multilayer structures using the Transfer Matrix Method.

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
        # make sure all keys are lowercase only
        args = {k.lower(): v for k, v in args.items()}
        # parse user inputs
        self.parse_input(args)
        # set refractive index array
        self.set_refractive_index_array()
        # compute reflectivity spectrum
        self.compute_spectrum()

        # this flag tells the code to determine the temperature
        # of an PV-STPV stack self-consistently
        # 0 means don't compute self-consistently
        self.loop_var = 0

        # print output message
        print(" Your spectra have been computed! \N{smiling face with sunglasses} ")

        if "therml" in args:
            args = {k.lower(): v for k, v in args.items()}
            self._parse_therml_input(args)
            self._compute_therml_spectrum(self.wavelength_array, self.emissivity_array)
            self._compute_power_density(self.wavelength_array)
            self._compute_stpv_power_density(self.wavelength_array)
            self._compute_stpv_spectral_efficiency(self.wavelength_array)
            self._compute_luminous_efficiency(self.wavelength_array)

            print(" Your therml spectra have been computed! \N{fire} ")

        # treat cooling specially because we need emissivity at lots of angles!
        if "cooling" in args:
            args = {k.lower(): v for k, v in args.items()}
            self._parse_therml_input(args)

            # get \epsilon_s(\lambda, \theta) and \epsilon_s(\lambda, \theta) for thermal radiation
            self.compute_explicit_angle_spectrum()
            print(
                " Your angle-dependent spectra have been computed! \N{smiling face with sunglasses} "
            )

            # call _compute_thermal_radiated_power( ) function
            self.radiative_cooling_power = self._compute_thermal_radiated_power(
                self.emissivity_array_s,
                self.emissivity_array_p,
                self.theta_vals,
                self.theta_weights,
                self.wavelength_array,
            )

            # call _compute_atmospheric_radiated_power() function
            self.atmospheric_warming_power = self._compute_atmospheric_radiated_power(
                self._atmospheric_transmissivity,
                self.emissivity_array_s,
                self.emissivity_array_p,
                self.theta_vals,
                self.theta_weights,
                self.wavelength_array,
            )

            # need to get one more set of \epsilon_s(\lambda, solar_angle) and \epsilon_p(\lamnda, solar_angle)
            self.incident_angle = self.solar_angle
            self.polarization = "s"
            self.compute_spectrum()
            solar_absorptivity_s = self.emissivity_array
            self.polarization = "p"
            self.compute_spectrum()
            solar_absorptivity_p = self.emissivity_array
            self.solar_warming_power = self._compute_solar_radiated_power(
                self._solar_spectrum,
                solar_absorptivity_s,
                solar_absorptivity_p,
                self.wavelength_array,
            )
            self.net_cooling_power = (
                self.radiative_cooling_power
                - self.atmospheric_warming_power
                - self.solar_warming_power
            )
            print(
                " Your radiative cooling quantities have been computed! \N{smiling face with sunglasses} "
            )

            # call _compute_solar_radiated_power() function

    def parse_input(self, args):
        """method to parse the user inputs and define structures / simulation
        Returns
        -------
        None
        """
        if "pv_bandgap_wavelength" in args:
            self.pv_lambda_bandgap = args["pv_bandgap_wavelength"]
        else:
            self.pv_lambda_bandgap = 700e-9

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
            self.thickness_array = np.array(args["thickness_list"])
        # default structure
        else:
            print("  Thickness array not specified!")
            print("  Proceeding with default structure - optically thick W! ")
            self.thickness_array = np.array([0, 900e-9, 0])

        if "material_list" in args:
            self.material_array = args["material_list"]
            self.number_of_layers = len(self.material_array)
        else:
            print("  Material array not specified!")
            print("  Proceeding with default structure - Air / SiO2 / Air ")
            self.material_array = ["Air", "SiO2", "Air"]
            self.number_of_layers = 3

        # user can specify which layers to compute gradients with respect to
        # i.e. for a structure like ['Air', 'SiO2', 'Ag', 'TiO2', 'Air]
        # the gradient list [1, 2] would take the gradients
        # with respect to layer 1 (top-most SiO2) and layer 2 (silver) only, while
        # leaving out layer 3 (TiO2)
        if "gradient_list" in args:
            self.gradient_list = np.array(args["gradient_list"])
        # default is all layers
        else:
            self.gradient_list = np.linspace(
                1, self.number_of_layers - 2, self.number_of_layers - 2, dtype=int
            )

        # if we specify a number of angles to compute the spectra over
        # this only gets used if we need explicit inclusion of angle of incidence/emission
        # for a given quantity
        if "number_of_angles" in args:
            self.number_of_angles = args["number_of_angles"]
        else:
            # this is a good default empirically if
            # Gauss-Legendre quadrature is used for angular spectra
            self.number_of_angles = 7

        # for now always get solar spectrum!
        self._solar_spectrum = self._read_AM()

        # for now always get atmospheric transmissivity spectru
        self._atmospheric_transmissivity = self._read_Atmospheric_Transmissivity()

    def set_refractive_index_array(self):
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
            # keep the original string too in case it is a file name
            _original_string = self.material_array[i]

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
            elif _lm == "ta2o5":
                self.material_Ta2O5(i)
            elif _lm == "tin":
                self.material_TiN(i)
            elif _lm == "tio2":
                self.material_TiO2(i)
            elif _lm == "w":
                self.material_W(i)
            # if we don't match one of these strings, then we assume the user has passed
            # a filename
            else:
                self.material_from_file(i, _original_string)

    def reverse_stack(self):
        """reverse the order of the stack
        e.g. if you have a structure that is Air/SiO2/HfO2/Ag/Air
        and you issue reverse_stack, the new structrure will be
        Air/Ag/HfO2/SiO2/Air
        """
        # store temporary versions of the RI array and thickness array
        _ri = self._refractive_index_array
        _ta = self.thickness_array

        # use np.flip to reverse the arrays
        self._refractive_index_array = np.flip(_ri, axis=1)
        self.thickness_array = np.flip(_ta)

    def remove_layer(self, layer_number):
        """remove layer number layer_number from your stack.
        e.g. if you have a structure that is Air/SiO2/HfO2/Ag/Air
        and you issue remove_layer(2), the new structrure will be
        Air/SiO2/Ag/Air
        """
        self.number_of_layers -= 1
        _nwl = len(self._refractive_index_array[:, 0])
        _nl = len(self._refractive_index_array[0, :])
        _temp_ri_array = np.copy(self._refractive_index_array)
        _temp_thickness_array = np.copy(self.thickness_array)

        _new_ri_array = np.zeros((_nwl, _nl - 1), dtype=complex)
        _new_thickness_array = np.zeros(_nl - 1)

        _new_ri_array[:, :layer_number] = _temp_ri_array[:, :layer_number]
        _new_thickness_array[:layer_number] = _temp_thickness_array[:layer_number]

        _new_ri_array[:, layer_number:] = _temp_ri_array[:, layer_number + 1 :]
        _new_thickness_array[layer_number:] = _temp_thickness_array[layer_number + 1 :]

        self._refractive_index_array = np.copy(_new_ri_array)
        self.thickness_array = np.copy(_new_thickness_array)

    def insert_layer(self, layer_number, layer_thickness):
        """insert an air layer between layer_number-1 and layer_number
        e.g. if you have a structure that is Air/SiO2/HfO2/Ag/Air
        and you issue insert_layer(1), the new structure will be
        Air/Air/SiO2/HfO2/Ag/Air
        if you issue insert_layer(2), the new structure will be
        Air/SiO2/Air/HfO2/Ag/Air

        """
        self.number_of_layers += 1
        _nwl = len(self._refractive_index_array[:, 0])
        _nl = len(self._refractive_index_array[0, :])
        _temp_ri_array = np.copy(self._refractive_index_array)
        _temp_thickness_array = np.copy(self.thickness_array)

        _new_ri_array = np.zeros((_nwl, _nl + 1), dtype=complex)
        _new_thickness_array = np.zeros(_nl + 1)
        _new_air_layer = np.ones(_nwl, dtype=complex) * 1.0

        _new_ri_array[:, :layer_number] = _temp_ri_array[:, :layer_number]
        _new_thickness_array[:layer_number] = _temp_thickness_array[:layer_number]

        _new_ri_array[:, layer_number] = _new_air_layer
        _new_thickness_array[layer_number] = layer_thickness

        _new_ri_array[:, layer_number + 1 :] = _temp_ri_array[:, layer_number:]
        _new_thickness_array[layer_number + 1 :] = _temp_thickness_array[layer_number:]

        self._refractive_index_array = np.copy(_new_ri_array)
        self.thickness_array = np.copy(_new_thickness_array)
        print(
            " A ",
            layer_thickness,
            " m air layer has been inserted into layer numbe ",
            layer_number,
        )
        print(
            " Use the `material_X(",
            layer_number,
            ") command to define the material of this new layer!",
        )

    def compute_spectrum(self):
        """computes the following attributes:
        Attributes
        ----------
        reflectivity_array : 1 x number_of_wavelengths numpy array of floats
            the reflectivity spectrum
        transmissivity_array : 1 x number_of_wavelengths numpy array of floats
            the transmissivity spectrum
        emissivity_array : 1 x number_of_wavelengths numpy array of floats
            the absorptivity / emissivity spectrum
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

            # if self.gradient==True:
            #    _tmg = self._compute_tm_grad(_ri, _k0, _kz, self.thickness_array)

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
        # self.render_color("ambient color")

    def compute_explicit_angle_spectrum(self):
        """computes the following attributes:
        Attributes
        ----------
        reflectivity_array_s : N_deg x number_of_wavelengths numpy array of floats
            the reflectivity spectrum vs wavelength and angle with s-polarization
        reflectivity_array_p : N_deg x number_of_wavelengths numpy array of floats
            the reflectivity spectrum vs wavelength and angle with p-polarization

        transmissivity_array_s : N_deg x number_of_wavelengths numpy array of floats
            the transmissivity spectrum vs wavelength and angle with s-polarization
        transmissivity_array_p : N_deg x number_of_wavelengths numpy array of floats
            the transmissivity spectrum vs wavelength and angle with p-polarization

        emissivity_array_s : N_deg x number_of_wavelengths numpy array of floats
            the emissivity spectrum vs wavelength and angle with s-polarization
        emissivity_array_p : N_deg x number_of_wavelengths numpy array of floats
            the emissivity spectrum vs wavelength and angle with p-polarization
        Returns
        -------
        None
        """
        # initialize the angle-dependent arrays
        self.reflectivity_array_s = np.zeros(
            (self.number_of_angles, self.number_of_wavelengths)
        )
        self.reflectivity_array_p = np.zeros(
            (self.number_of_angles, self.number_of_wavelengths)
        )

        self.transmissivity_array_s = np.zeros(
            (self.number_of_angles, self.number_of_wavelengths)
        )
        self.transmissivity_array_p = np.zeros(
            (self.number_of_angles, self.number_of_wavelengths)
        )

        self.emissivity_array_s = np.zeros(
            (self.number_of_angles, self.number_of_wavelengths)
        )
        self.emissivity_array_p = np.zeros(
            (self.number_of_angles, self.number_of_wavelengths)
        )

        # now set up the angular Gauss-Legendre grid
        a = 0
        b = np.pi / 2.0
        self.x, self.theta_weights = np.polynomial.legendre.leggauss(
            self.number_of_angles
        )
        self.theta_vals = 0.5 * (self.x + 1) * (b - a) + a
        self.theta_weights = self.theta_weights * 0.5 * (b - a)

        # compute k0 which does not care about angle
        self._compute_k0()

        # loop over angles first
        for i in range(0, self.number_of_angles):
            self.incident_angle = self.theta_vals[i]
            # compute kx and kz, which do depend on angle
            self._compute_kx()
            self._compute_kz()

            for j in range(0, self.number_of_wavelengths):
                _k0 = self._k0_array[j]
                _ri = self._refractive_index_array[j, :]
                _kz = self._kz_array[j, :]

                # get transfer matrix, theta_array, and co_theta_array for current k0 value and 's' polarization
                self.polarization = "s"
                _tm_s, _theta_array_s, _cos_theta_array_s = self._compute_tm(
                    _ri, _k0, _kz, self.thickness_array
                )

                # get transfer matrix, theta_array, and cos_theta_array for current k0 value and 'p' polarization
                self.polarization = "p"
                _tm_p, _theta_array_p, _cos_theta_array_p = self._compute_tm(
                    _ri, _k0, _kz, self.thickness_array
                )

                # reflection amplitude
                _r_s = _tm_s[1, 0] / _tm_s[0, 0]
                _r_p = _tm_p[1, 0] / _tm_p[0, 0]

                # transmission amplitude
                _t_s = 1 / _tm_s[0, 0]
                _t_p = 1 / _tm_p[0, 0]

                # refraction angle and RI prefractor for computing transmission
                _factor_s = (
                    _ri[self.number_of_layers - 1]
                    * _cos_theta_array_s[self.number_of_layers - 1]
                    / (_ri[0] * _cos_theta_array_s[0])
                )
                _factor_p = (
                    _ri[self.number_of_layers - 1]
                    * _cos_theta_array_p[self.number_of_layers - 1]
                    / (_ri[0] * _cos_theta_array_p[0])
                )

                # reflectivity
                self.reflectivity_array_s[i, j] = np.real(_r_s * np.conj(_r_s))
                self.reflectivity_array_p[i, j] = np.real(_r_p * np.conj(_r_p))

                # transmissivity
                self.transmissivity_array_s[i, j] = np.real(
                    _t_s * np.conj(_t_s) * _factor_s
                )
                self.transmissivity_array_p[i, j] = np.real(
                    _t_p * np.conj(_t_p) * _factor_p
                )

                # emissivity
                self.emissivity_array_s[i, j] = (
                    1
                    - self.reflectivity_array_s[i, j]
                    - self.transmissivity_array_s[i, j]
                )
                self.emissivity_array_p[i, j] = (
                    1
                    - self.reflectivity_array_p[i, j]
                    - self.transmissivity_array_p[i, j]
                )

    def compute_spectrum_gradient(self):
        """computes the following attributes:
        Attributes
        ----------
        reflectivity_gradient_array : number_of_wavelengths x len(gradient_list) numpy array of floats
            the reflectivity spectrum
        transmissivity_gradient_array : number_of_wavelengths x len(gradient_list) numpy array of floats
            the transmissivity spectrum
        emissivity_gradient_array : number_of_wavelengths x len(gradient_list) numpy array of floats
            the absorptivity / emissivity spectrum
        Returns
        -------
        None
        """

        # initialize gradient arrays
        # _nwl -> number of wavelengths
        _nwl = len(self.wavelength_array)
        # _ngr -> number of gradient dimensions
        _ngr = len(self.gradient_list)

        self.reflectivity_gradient_array = np.zeros((_nwl, _ngr))
        self.transmissivity_gradient_array = np.zeros((_nwl, _ngr))
        self.emissivity_gradient_array = np.zeros((_nwl, _ngr))

        for i in range(0, _ngr):
            for j in range(0, _nwl):
                _k0 = self._k0_array[j]
                _ri = self._refractive_index_array[j, :]
                _kz = self._kz_array[j, :]

                # get transfer matrix, theta_array, and co_theta_array for current k0 value
                _tm, _theta_array, _cos_theta_array = self._compute_tm(
                    _ri, _k0, _kz, self.thickness_array
                )

                # get gradient of transfer matrix with respect to layer i
                _tm_grad, _theta_array, _cos_theta_array = self._compute_tm_gradient(
                    _ri, _k0, _kz, self.thickness_array, i + 1
                )

                # Using equation (14) from https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.013018
                # for the derivative of the reflection amplitude for wavelength j with respect to layer i
                # from wptherml: r_prime = (M11*M21p[j] - M21*M11p[j])/(M11*M11)
                r_prime = (_tm[0, 0] * _tm_grad[1, 0] - _tm[1, 0] * _tm_grad[0, 0]) / (
                    _tm[0, 0] ** 2
                )
                # using equation (12) to get the reflection amplitude at wavelength j
                r = _tm[1, 0] / _tm[0, 0]

                # Using equation (10) to get the derivative of R at waveleength j with respect to layer i
                self.reflectivity_gradient_array[j, i] = np.real(
                    r_prime * np.conj(r) + r * np.conj(r_prime)
                )

                # compute t_prime using equation (15)
                t_prime = -_tm_grad[0, 0] / _tm[0, 0] ** 2

                # compute t using equation equation (13)
                t = 1 / _tm[0, 0]

                _factor = (
                    _ri[self.number_of_layers - 1]
                    * _cos_theta_array[self.number_of_layers - 1]
                    / (_ri[0] * _cos_theta_array[0])
                )

                # compute the derivative of T at wavelength j with respect to layer i using Eq. (11)
                self.transmissivity_gradient_array[j, i] = np.real(
                    (t_prime * np.conj(t) + t * np.conj(t_prime)) * _factor
                )
                # derivative of \epsilon is - \partial R / \partial s -\partial T / \partial sß
                self.emissivity_gradient_array[j, i] = (
                    -self.transmissivity_gradient_array[j, i]
                    - self.reflectivity_gradient_array[j, i]
                )

    def compute_explicit_angle_spectrum_gradient(self):
        """computes the following attributes:
        Attributes
        ----------
        reflectivity_gradient_array_s : N_deg x number_of_wavelengths x len(gradient_list) numpy array of floats
            the gradient of the s-polarized reflectivity spectrum vs angle

        reflectivity_gradient_array_p : N_deg x number_of_wavelengths x len(gradient_list) numpy array of floats
            the gradient of the p-polarized reflectivity spectrum vs angle

        transmissivity_gradient_array_s : N_deg x number_of_wavelengths x len(gradient_list) numpy array of floats
            the grdient of the s-polarized transmissivity spectrum vs angle

        transmissivity_gradient_array_p : N_deg x number_of_wavelengths x len(gradient_list) numpy array of floats
            the grdient of the p-polarized transmissivity spectrum vs angle

        emissivity_gradient_array_s : N_deg x number_of_wavelengths x len(gradient_list) numpy array of floats
            the grdient of the s-polarized emissivity spectrum vs angle

        emissivity_gradient_array_p : N_deg x number_of_wavelengths x len(gradient_list) numpy array of floats
            the grdient of the p-polarized emissivity spectrum vs angle

        Returns
        -------
        None
        """
        # initialize gradient arrays
        # _nwl -> number of wavelengths
        _nwl = len(self.wavelength_array)
        # _ngr -> number of gradient dimensions
        _ngr = len(self.gradient_list)
        # _nth -> number of angles
        _nth = self.number_of_angles

        # jjf note - _nwl is going to be the longest axis in most cases
        # should it be either the inner-most or outter-most dimension instead for
        # performance reasons?
        self.reflectivity_gradient_array_s = np.zeros((_nth, _nwl, _ngr))
        self.reflectivity_gradient_array_p = np.zeros((_nth, _nwl, _ngr))

        self.transmissivity_gradient_array_s = np.zeros((_nth, _nwl, _ngr))
        self.transmissivity_gradient_array_p = np.zeros((_nth, _nwl, _ngr))

        self.emissivity_gradient_array_s = np.zeros((_nth, _nwl, _ngr))
        self.emissivity_gradient_array_p = np.zeros((_nth, _nwl, _ngr))

        # compute k0 which does not care about angle
        self._compute_k0()

        # loop over angles first
        for k in range(0, _nth):
            self.incident_angle = self.theta_vals[k]
            # compute kx and kz, which do depend on angle
            self._compute_kx()
            self._compute_kz()

            for i in range(0, _ngr):
                for j in range(0, _nwl):
                    _k0 = self._k0_array[j]
                    _ri = self._refractive_index_array[j, :]
                    _kz = self._kz_array[j, :]

                    # s-polarization first
                    self.polarization = "s"
                    # get transfer matrix, theta_array, and co_theta_array for current k0 value
                    _tm, _theta_array, _cos_theta_array = self._compute_tm(
                        _ri, _k0, _kz, self.thickness_array
                    )

                    # get gradient of transfer matrix with respect to layer i
                    (
                        _tm_grad,
                        _theta_array,
                        _cos_theta_array,
                    ) = self._compute_tm_gradient(
                        _ri, _k0, _kz, self.thickness_array, i + 1
                    )

                    # Using equation (14) from https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.013018
                    # for the derivative of the reflection amplitude for wavelength j with respect to layer i
                    # from wptherml: r_prime = (M11*M21p[j] - M21*M11p[j])/(M11*M11)
                    r_prime = (
                        _tm[0, 0] * _tm_grad[1, 0] - _tm[1, 0] * _tm_grad[0, 0]
                    ) / (_tm[0, 0] ** 2)
                    # using equation (12) to get the reflection amplitude at wavelength j
                    r = _tm[1, 0] / _tm[0, 0]

                    # Using equation (10) to get the derivative of R at waveleength j with respect to layer i
                    self.reflectivity_gradient_array_s[k, j, i] = np.real(
                        r_prime * np.conj(r) + r * np.conj(r_prime)
                    )

                    # compute t_prime using equation (15)
                    t_prime = -_tm_grad[0, 0] / _tm[0, 0] ** 2

                    # compute t using equation equation (13)
                    t = 1 / _tm[0, 0]

                    _factor = (
                        _ri[self.number_of_layers - 1]
                        * _cos_theta_array[self.number_of_layers - 1]
                        / (_ri[0] * _cos_theta_array[0])
                    )

                    # compute the derivative of T at wavelength j with respect to layer i using Eq. (11)
                    self.transmissivity_gradient_array_s[k, j, i] = np.real(
                        (t_prime * np.conj(t) + t * np.conj(t_prime)) * _factor
                    )
                    # derivative of \epsilon is - \partial R / \partial s -\partial T / \partial sß
                    self.emissivity_gradient_array_s[k, j, i] = (
                        -self.transmissivity_gradient_array_s[k, j, i]
                        - self.reflectivity_gradient_array_s[k, j, i]
                    )

                    # p-polarization second
                    self.polarization = "p"
                    # get transfer matrix, theta_array, and co_theta_array for current k0 value
                    _tm, _theta_array, _cos_theta_array = self._compute_tm(
                        _ri, _k0, _kz, self.thickness_array
                    )

                    # get gradient of transfer matrix with respect to layer i
                    (
                        _tm_grad,
                        _theta_array,
                        _cos_theta_array,
                    ) = self._compute_tm_gradient(
                        _ri, _k0, _kz, self.thickness_array, i + 1
                    )

                    # Using equation (14) from https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.013018
                    # for the derivative of the reflection amplitude for wavelength j with respect to layer i
                    # from wptherml: r_prime = (M11*M21p[j] - M21*M11p[j])/(M11*M11)
                    r_prime = (
                        _tm[0, 0] * _tm_grad[1, 0] - _tm[1, 0] * _tm_grad[0, 0]
                    ) / (_tm[0, 0] ** 2)
                    # using equation (12) to get the reflection amplitude at wavelength j
                    r = _tm[1, 0] / _tm[0, 0]

                    # Using equation (10) to get the derivative of R at waveleength j with respect to layer i
                    self.reflectivity_gradient_array_p[k, j, i] = np.real(
                        r_prime * np.conj(r) + r * np.conj(r_prime)
                    )

                    # compute t_prime using equation (15)
                    t_prime = -_tm_grad[0, 0] / _tm[0, 0] ** 2

                    # compute t using equation equation (13)
                    t = 1 / _tm[0, 0]

                    _factor = (
                        _ri[self.number_of_layers - 1]
                        * _cos_theta_array[self.number_of_layers - 1]
                        / (_ri[0] * _cos_theta_array[0])
                    )

                    # compute the derivative of T at wavelength j with respect to layer i using Eq. (11)
                    self.transmissivity_gradient_array_p[k, j, i] = np.real(
                        (t_prime * np.conj(t) + t * np.conj(t_prime)) * _factor
                    )
                    # derivative of \epsilon is - \partial R / \partial s -\partial T / \partial sß
                    self.emissivity_gradient_array_p[k, j, i] = (
                        -self.transmissivity_gradient_array_p[k, j, i]
                        - self.reflectivity_gradient_array_p[k, j, i]
                    )

    def compute_stpv(self):
        """compute the figures of merit for STPV applications, including"""
        self.compute_spectrum()
        self._compute_therml_spectrum(self.wavelength_array, self.emissivity_array)
        self._compute_power_density(self.wavelength_array)
        self._compute_stpv_power_density(self.wavelength_array)
        self._compute_stpv_spectral_efficiency(self.wavelength_array)

    def compute_stpv_gradient(self):
        self.compute_spectrum()
        self.compute_spectrum_gradient()
        self._compute_therml_spectrum_gradient(
            self.wavelength_array, self.emissivity_gradient_array
        )
        self._compute_power_density_gradient(self.wavelength_array)
        self._compute_stpv_power_density_gradient(self.wavelength_array)
        self._compute_stpv_spectral_efficiency_gradient(self.wavelength_array)

    def compute_pv_stpv_jsc(self):
        """
        Function to compute the f_C figure of merit for pv-stpv, Eq. (45) here: https://www.overleaf.com/project/648a0cfeae29e31e10afc075
        We will assume the base layer is the AR + Polystyrene stack so we
        will add the PSC layer here too
        """
        # First make sure we have full stack including the PSC layer
        # get terminal layer number
        _ln = len(self.thickness_array) - 1
        # insert thick active layer as the bottom-most layer
        self.insert_layer(_ln, 1000e-9)
        # make sure the active layer has RI of 2D perovskite
        self.material_2D_HOIP(_ln)
        self.compute_spectrum()
        absorptivity_full_stack = self.emissivity_array
        
        # get envelope function that behaves like ideal spectral response function
        bg_idx = np.abs(self.wavelength_array - self.pv_lambda_bandgap).argmin()
        env = np.zeros_like(self.wavelength_array)
        env[:bg_idx] = self.wavelength_array[:bg_idx] / self.pv_lambda_bandgap

        # scale AM by \lambda / \lambda_bg
        # compute the useful power density spectrum
        power_density_array = (
            self._solar_spectrum * absorptivity_full_stack * env
        )

        self.pv_stpv_jsc = np.trapz(
            power_density_array, self.wavelength_array
        )

    def compute_pv_stpv(self):
        """Turn this into a proper docstring

            Let's denote the PV-STPV stack as

            ---------------------------------
                layer A (AR + polystyrene)
            ---------------------------------
                layer B (active perovskite)
            ---------------------------------

        where we need to determine the amount of solar light absorbed into
        the active layer (layer B) and we need to also compute the thermal
        emission spectrum of layer A into layer B.

        Upon entering the compute_pv_stpv function, layer A will be defined.  We
        can therefore compute the "top-side" (relevant for solar absorption)
        absorptivity of layer A alone, as well as the "bottom-side" emissivity
        (relevant for thermal emission into layer B) of layer A alone. We will
        need to reverse the layer A structure to get the "bottom-side" emissivity.

        Top-side absorptivity of layer A: absorptivity_A_T
        Bottom-side absorptivity of layer A: emissivity_A_B
        Top-side transmissivity of layer A: transmissivity_A_T
        Top-side reflectivity of layer A: reflectivity_A_T

        After these quantities are calculated, we can reverse the structure once again so
        that it matches the original order of layer A and add Layer B to the bottom.
        At this point, we can compute the "top-side" absorptivity of layer AB:
        absorptivity_AB_T and then approximate the "top-side" absorptivity of
        the actual active layer (layer B) as
        absororptivity_B_T = absorptivity_AB_T - absorptivity_A_T
        This is an approximation, and we should figure out how to do this rigorously
        using the transfer matrix!

        """
        # temporarily set the temperature to 440 K
        _T = self.temperature
        self.temperature = 440
        # get the transmissivity of the stack and get transmitted solar spectrum
        self.compute_spectrum()
        absorptivity_A_T = self.emissivity_array
        transmissivity_A_T = self.transmissivity_array
        reflectivity_A_T = self.reflectivity_array

        # make sure we have the solar spectrum
        self._solar_spectrum = self._read_AM()

        # define transmitted solar spectrum
        self.transmitted_solar_spectrum = self._solar_spectrum * transmissivity_A_T
        self.pv_stpv_transmitted_solar_power = np.trapz(
            self.transmitted_solar_spectrum, self.wavelength_array
        )

        # reverse stack and get thermal emission spectrum of the stack INTO the active layer
        self.reverse_stack()
        self.compute_spectrum()
        emissivity_A_B = self.emissivity_array

        # get Blackbody spectrum at the default temperature - this is tentative
        self._compute_therml_spectrum(self.wavelength_array, emissivity_A_B)
        self.pv_stpv_p_split = np.pi * np.trapz(
            self.thermal_emission_array, self.wavelength_array
        )

        # now add perovskite layer to the stack and get the emissivity/absorptivity towards the sky
        self.reverse_stack()

        # get terminal layer number
        _ln = len(self.thickness_array) - 1
        # insert thick active layer as the bottom-most layer
        self.insert_layer(_ln, 1000e-9)
        # make sure the active layer has RI of 2D perovskite
        self.material_2D_HOIP(_ln)
        self.compute_spectrum()
        absorptivity_AB_T = self.emissivity_array

        # approximate the absorptivity only of the active layer
        absorptivity_B_T = absorptivity_AB_T - absorptivity_A_T

        # get the absorbed power
        self.pv_stpv_p_abs = np.trapz(
            absorptivity_B_T * self._solar_spectrum, self.wavelength_array
        )

        # loop over temperature to try to find the temperature of the stack that balances emitted
        # power with absorbed power
    
        if self.loop_var==1:
            _kill = 1
            _T = 300
    
            while(_kill):
                _bbs = self._compute_blackbody_spectrum(self.wavelength_array, _T)
                P_emit = np.trapz( np.pi/2 * _bbs * (emissivity_A_B + absorptivity_AB_T), self.wavelength_array)
                _T += 1
                if P_emit > absorptivity_B_T :
                    _kill = 0

        else:
            _T = 440

        self._compute_pv_stpv_power_density(self.wavelength_array)
        # reverse stack again and add active layer and get absorbed power into the structure
        self.reverse_stack()

        # approximate ideal spectral response using pv_lambda_bandgap
        self.spectral_response = self.wavelength_array / self.pv_lambda_bandgap
        # now compute pv_stpv short circuit current
        self._compute_pv_short_circuit_current(
            self.wavelength_array,
            self.emissivity_array,
            self.spectral_response,
            self._solar_spectrum,
        )
        self.remove_layer(_ln)

        # reset temperature to whatever it was at the beginning
        self.temperature = _T
    
    # Other figure of merit calculations here
    # Pulling functions from compute_pv_stpv on their own (to be called in compute_pv_stpv)

    def compute_spectral_response(self)
        """Docstring
        
        """
        # Figure of Merit one


    def compute_pv_stpv_total_incident_power(self):
        """Docstring
        Use equation npv = Jsc * Voc * FF
        Jsc = short circuit current
        Voc = open circuit current
        FF = fill factor
        
        The plan:
        initialze npv (assuming a static number), calculate Voc and FF, and multiply these three together to get total incident power as a unitless efficiency 

        total_incident_power = pv_stpv_short_circuit_current_gradient * Voc
        
        """
        # Figure of Merit two


    def compute_pv_stpv_gradient(self):
        """
        Computes the following attributes for short circuit current calculation:

        Attributes
        ----------

        e_gradient_index : Integer
                        Length of the emissivity gradient array.

        emissivity_gradient_array_prime : Array
                                        (Emissivity gradient array x Wavelength array) / Lambda bandgap.

        pv_stpv_short_circuit_current : Float
                                    Short circuit current as defined in Equation (23) of https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.013018
                                     the integration of Emissivity x Spectral Response x Solar Spectrum over wavelength.

        Returns:
        --------
        None
        """

        # Looking at the short circuit current (Jsc)
        # Need to iterate over emissivity_gradient_array for every value at a given wavelength.
        # Need to take the integral of this multiplied by _solar_spectrum and spectral_response (both precalculated), between 0 and lambda bandgap.

        # Acquire necessary variables
        self._solar_spectrum = self._read_AM()
        self.compute_spectrum_gradient()

        # Initialize short circuit current array
        e_gradient_index = len(self.emissivity_gradient_array[0, :])
        self.pv_stpv_short_circuit_current_gradient = np.zeros(e_gradient_index)

        bg_idx = np.abs(self.wavelength_array - self.pv_lambda_bandgap).argmin()
        _spectral_response = np.zeros_like(self.wavelength_array)

        _spectral_response[:bg_idx] = self.wavelength_array[:bg_idx] / self.pv_lambda_bandgap

        # Iterate over material thicknesses
        for i in range(0, e_gradient_index):
            self.pv_stpv_short_circuit_current_gradient[i] = np.trapz(
                self.emissivity_gradient_array[:,i]
                * _spectral_response
                * self._solar_spectrum,
                self.wavelength_array,
            )  # Integrate for short circuit current


    def compute_cooling(self):
        """Method to compute the radiative cooling figures of merit

        Attributes
        ----------
        self.radiative_cooling_power : float
            the thermal power radiated by a structure into the universe (P_rad)

        self.atmospheric_warming_power : float
            the thermal power radiated by the atmosphere and absorbed by a structure (P_atm)

        self.solar_warming_power : float
            the thermal power radiated by the sun and absorbed by a structure (P_sun)

        self.net_cooling_power : float
            P_rad - P_atm - P_sum ... if positive, the structure is net cooling if negative, it is net heating

        Returns
        -------
        None
        """

        # get \epsilon_s(\lambda, \theta) and \epsilon_s(\lambda, \theta) for thermal radiation
        self.compute_explicit_angle_spectrum()

        # call _compute_thermal_radiated_power( ) function
        self.radiative_cooling_power = self._compute_thermal_radiated_power(
            self.emissivity_array_s,
            self.emissivity_array_p,
            self.theta_vals,
            self.theta_weights,
            self.wavelength_array,
        )

        # call _compute_atmospheric_radiated_power() function
        self.atmospheric_warming_power = self._compute_atmospheric_radiated_power(
            self._atmospheric_transmissivity,
            self.emissivity_array_s,
            self.emissivity_array_p,
            self.theta_vals,
            self.theta_weights,
            self.wavelength_array,
        )

        # need to get one more set of \epsilon_s(\lambda, solar_angle) and \epsilon_p(\lamnda, solar_angle)
        self.incident_angle = self.solar_angle
        self.polarization = "s"
        self.compute_spectrum()
        solar_absorptivity_s = self.emissivity_array
        self.polarization = "p"
        self.compute_spectrum()
        solar_absorptivity_p = self.emissivity_array
        self.solar_warming_power = self._compute_solar_radiated_power(
            self._solar_spectrum,
            solar_absorptivity_s,
            solar_absorptivity_p,
            self.wavelength_array,
        )
        self.net_cooling_power = (
            self.radiative_cooling_power
            - self.solar_warming_power
            - self.atmospheric_warming_power
        )

    def compute_cooling_gradient(self):
        # get the gradient of the emissivity vs angle and wavelength
        self.compute_explicit_angle_spectrum_gradient()
        self.radiative_cooling_power_gradient = (
            self._compute_thermal_radiated_power_gradient(
                self.emissivity_gradient_array_s,
                self.emissivity_gradient_array_p,
                self.theta_vals,
                self.theta_weights,
                self.wavelength_array,
            )
        )

        self.atmospheric_warming_power_gradient = (
            self._compute_atmospheric_radiated_power_gradient(
                self._atmospheric_transmissivity,
                self.emissivity_gradient_array_s,
                self.emissivity_gradient_array_p,
                self.theta_vals,
                self.theta_weights,
                self.wavelength_array,
            )
        )

        # need to get one more set of \epsilon_s(\lambda, solar_angle) and \epsilon_p(\lamnda, solar_angle)
        self.incident_angle = self.solar_angle
        self.polarization = "s"
        self.compute_spectrum()
        self.compute_spectrum_gradient()
        solar_absorptivity_s = self.emissivity_gradient_array
        self.polarization = "p"
        self.compute_spectrum()
        self.compute_spectrum_gradient()
        solar_absorptivity_p = self.emissivity_gradient_array

        self.solar_warming_power_gradient = self._compute_solar_radiated_power_gradient(
            self._solar_spectrum,
            solar_absorptivity_s,
            solar_absorptivity_p,
            self.wavelength_array,
        )
        self.net_cooling_power_gradient = (
            self.radiative_cooling_power_gradient
            - self.solar_warming_power_gradient
            - self.atmospheric_warming_power_gradient
        )

    def _compute_kz(self):
        """computes the z-component of the wavevector in each layer of the stack
        Attributes
        ----------
            _refractive_index_array : number_of_wavelength x number_of_layers numpy array of complex floats
                the array of refractive index values corresponding to wavelength_array
            _kz_array : number_of_wavelength x number_of_layers numpy array of complex floats
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

    def _compute_tm_gradient(self, _refractive_index, _k0, _kz, _d, _ln):
        """compute the transfer matrix for each wavelength
        _ln : int
            specifies the layer number the gradient will be taken with respect to
        Returns
        -------
        _tm_gradient : 2 x 2 complex numpy array
            transfer matrix for the _k0 value
        _THETA : 1 x number_of_layers complex numpy array
            refraction angles in each layer for the _k0 value
        _CTHETA : 1 x number_of_layers complex numpy array
            cosine of the refraction angles in each layer for the _k0 value
        JJF Note: Basically the only difference between the calculation
        of the dM/dS_ln and M is that a single P matrix corresponding
        to _ln is replaced by dP/DP_ln.  So, you can basically modify the
        loop where _PM is computed to have a conditional that
        computes _PM by calling _compute_pm_gradient instead of _compute_pm
        when i == _ln
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
        # initialize _tm_gradient here!  (was previously _tm)
        _DM[:, :, 0], _tm_gradient = self._compute_dm(_refractive_index[0], _CTHETA[0])

        for i in range(1, self.number_of_layers - 1):
            _DM[:, :, i], _DIM[:, :, i] = self._compute_dm(
                _refractive_index[i], _CTHETA[i]
            )
            if i == _ln:
                _PM[:, :, i] = self._compute_pm_analytical_gradient(_kz[i], _PHIL[i])
            else:
                _PM[:, :, i] = self._compute_pm(_PHIL[i])

            _tm_gradient = np.matmul(_tm_gradient, _DM[:, :, i])
            _tm_gradient = np.matmul(_tm_gradient, _PM[:, :, i])
            _tm_gradient = np.matmul(_tm_gradient, _DIM[:, :, i])

        (
            _DM[:, :, self.number_of_layers - 1],
            _DIM[:, :, self.number_of_layers - 1],
        ) = self._compute_dm(
            _refractive_index[self.number_of_layers - 1],
            _CTHETA[self.number_of_layers - 1],
        )

        _tm_gradient = np.matmul(_tm_gradient, _DM[:, :, self.number_of_layers - 1])

        return _tm_gradient, _THETA, _CTHETA

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

    def _compute_pm_analytical_gradient(self, kzl, phil):
        """compute the derivative of the P matrix with respect to layer thickness

        Arguments
        ---------
            kzl : complex float
                the z-component of the wavevector in layer l
            phil : complex float
                kzl * sl where sl is the thickness of layer l
        Reference
        ---------
            Equation 18 of https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.2.013018
        Returns
        -------
            _pm_analytical_gradient : 2x2 numpy array of complex floats
                the analytical derivative of the P matrix with respect to thickness of layer l

        """
        _pm_analytical_gradient = np.zeros((2, 2), dtype=complex)
        _ci = 0 + 1j
        _a = -1 * _ci * phil
        _b = _ci * phil

        _pm_analytical_gradient[0, 0] = -_ci * kzl * np.exp(_a)
        _pm_analytical_gradient[1, 1] = _ci * kzl * np.exp(_b)

        return _pm_analytical_gradient

    def _compute_rgb(self, colorblindness="False"):
        # get color response functions
        self._read_CIE()
        # plt.plot(self.wavelength_array * 1e9, self.reflectivity_array, label="Reflectivity")
        # plt.plot(self.wavelength_array * 1e9, self._cie_cr, label="CIE Red")
        # plt.legend()
        # plt.plot(self.wavelength_array, self._cie_cr * self.reflectivity_array)
        # plt.show()

        # get X, Y, and Z from reflectivity spectrum and Cr, Cg, Cb response functions
        X = np.trapz(self._cie_cr * self.reflectivity_array, self.wavelength_array)
        Y = np.trapz(self._cie_cg * self.reflectivity_array, self.wavelength_array)
        Z = np.trapz(self._cie_cb * self.reflectivity_array, self.wavelength_array)

        # zero out appropriate response if colorblindness is indicated
        # from here: https://www.color-blindness.com/types-of-color-blindness/
        # Tritanopia/Tritanomaly: Missing/malfunctioning S-cone (blue).
        # Deuteranopia/Deuteranomaly: Missing/malfunctioning M-cone (green).
        # Protanopia/Protanomaly: Missing/malfunctioning L-cone (red).

        if colorblindness == "Tritanopia" or colorblindness == "Tritanomaly":
            Z = 0
        if colorblindness == "Deuteranopia" or colorblindness == "Deuteranomaly":
            Y = 0
        if colorblindness == "Protanopia" or colorblindness == "Protanomaly":
            X = 0

        # get total magnitude
        tot = X + Y + Z

        # get normalized values
        x = X / tot
        y = Y / tot
        z = Z / tot

        # should also be equal to z = 1 - x - y
        # array of xr, xg, xb, xw, ..., zr, zg, zb, zw
        # use hdtv standard
        xrgbw = [0.670, 0.210, 0.150, 0.3127]
        yrgbw = [0.330, 0.710, 0.060, 0.3291]
        zrgbw = []
        for i in range(0, len(xrgbw)):
            zrgbw.append(1.0 - xrgbw[i] - yrgbw[i])

        ## rx = yg*zb - yb*zg
        rx = yrgbw[1] * zrgbw[2] - yrgbw[2] * zrgbw[1]
        ## ry = xb*zg - xg*zb
        ry = xrgbw[2] * zrgbw[1] - xrgbw[1] * zrgbw[2]
        ## rz = (xg * yb) - (xb * yg)
        rz = xrgbw[1] * yrgbw[2] - xrgbw[2] * yrgbw[1]
        ## gx = (yb * zr) - (yr * zb)
        gx = yrgbw[2] * zrgbw[0] - yrgbw[0] * zrgbw[2]
        ## gy = (xr * zb) - (xb * zr)
        gy = xrgbw[0] * zrgbw[2] - xrgbw[2] * zrgbw[0]
        ## gz = (xb * yr) - (xr * yb)
        gz = xrgbw[2] * yrgbw[0] - xrgbw[0] * yrgbw[2]
        ## bx = (yr * zg) - (yg * zr)
        bx = yrgbw[0] * zrgbw[1] - yrgbw[1] * zrgbw[0]
        ## by = (xg * zr) - (xr * zg)
        by = xrgbw[1] * zrgbw[0] - xrgbw[0] * zrgbw[1]
        ## bz = (xr * yg) - (xg * yr)
        bz = xrgbw[0] * yrgbw[1] - xrgbw[1] * yrgbw[0]

        ## rw = ((rx * xw) + (ry * yw) + (rz * zw)) / yw;
        rw = (rx * xrgbw[3] + ry * yrgbw[3] + rz * zrgbw[3]) / yrgbw[3]
        ## gw = ((gx * xw) + (gy * yw) + (gz * zw)) / yw;
        gw = (gx * xrgbw[3] + gy * yrgbw[3] + gz * zrgbw[3]) / yrgbw[3]
        ## bw = ((bx * xw) + (by * yw) + (bz * zw)) / yw;
        bw = (bx * xrgbw[3] + by * yrgbw[3] + bz * zrgbw[3]) / yrgbw[3]

        ## /* xyz -> rgb matrix, correctly scaled to white. */
        rx = rx / rw
        ry = ry / rw
        rz = rz / rw
        gx = gx / gw
        gy = gy / gw
        gz = gz / gw
        bx = bx / bw
        by = by / bw
        bz = bz / bw

        ## /* rgb of the desired point */
        r = (rx * x) + (ry * y) + (rz * z)
        g = (gx * x) + (gy * y) + (gz * z)
        b = (bx * x) + (by * y) + (bz * z)

        rgblist = []
        rgblist.append(r)
        rgblist.append(g)
        rgblist.append(b)

        # are there negative values?
        w = np.amin(rgblist)
        if w < 0:
            rgblist[0] = rgblist[0] - w
            rgblist[1] = rgblist[1] - w
            rgblist[2] = rgblist[2] - w

        # scale things so that max has value of 1
        mag = np.amax(rgblist)

        rgblist[0] = rgblist[0] / mag
        rgblist[1] = rgblist[1] / mag
        rgblist[2] = rgblist[2] / mag

        # rgb = {'r': rgblist[0]/mag, 'g': rgblist[1]/mag, 'b': rgblist[2]/mag }

        return rgblist

    def render_color(self, string, colorblindness="False"):
        fig, ax = plt.subplots()
        # The grid of visible wavelengths corresponding to the grid of colour-matching
        # functions used by the ColourSystem instance.

        # Calculate the black body spectrum and the HTML hex RGB colour string
        cierbg = self._compute_rgb(colorblindness)
        # cierbg = [1.,0.427,0.713]
        # Place and label a circle with the colour of a black body at temperature T
        x, y = 0.0, 0.0
        circle = Circle(xy=(x, y * 1.2), radius=0.4, fc=cierbg)
        ax.add_patch(circle)
        ax.annotate(
            string, xy=(x, y * 1.2 - 0.5), va="center", ha="center", color=cierbg
        )

        # Set the limits and background colour; remove the ticks
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("k")
        # ax.set_axis_bgcolor('k')
        # Make sure our circles are circular!
        ax.set_aspect("equal")
        plt.show()
