from .spectrum_driver import SpectrumDriver
from .materials import Materials
from .therml import Therml
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy.linalg.blas import zgemm  # BLAS-optimized complex matrix multiplication
import logging

logger = logging.getLogger(__name__)

#@jit(nopython=True)
def _compute_dm(refractive_index, cosine_theta, polarization):
    """Compute the D and D_inv matrices for each layer and wavelength"""
    _dm = np.zeros((2, 2), dtype=np.complex128)
    _dim = np.zeros((2, 2), dtype=np.complex128)

    if polarization == "s":
        _dm[0, 0] = 1
        _dm[0, 1] = 1
        _dm[1, 0] = refractive_index * cosine_theta
        _dm[1, 1] = -refractive_index * cosine_theta

    elif polarization == "p":
        _dm[0, 0] = cosine_theta
        _dm[0, 1] = cosine_theta
        _dm[1, 0] = refractive_index
        _dm[1, 1] = -refractive_index

    # Faster manual inversion of 2x2 matrix
    _det = 1 / (_dm[0, 0] * _dm[1, 1] - _dm[0, 1] * _dm[1, 0])
    _dim[0, 0] = _det * _dm[1, 1]
    _dim[0, 1] = -_det * _dm[0, 1]
    _dim[1, 0] = -_det * _dm[1, 0]
    _dim[1, 1] = _det * _dm[0, 0]

    return _dm, _dim

def compute_p_matrices_vectorized(phil):
    """
    Computes P matrices for each wavelength and layer using vectorized operations.
    Args:
        phil: NumPy array (Nl, Nd) containing phil values.
    Returns:
        NumPy array (Nl, Nd - 2, 2, 2) containing P matrices.
    """
    Nl, Nd = phil.shape
    P_matrices = np.zeros((Nl, Nd - 2, 2, 2), dtype=np.complex128)
    _ci = 1j
    # Slice phil to exclude the first and last layers
    phil_intermediate = phil[:, 1:-1]
    # Compute exponential terms
    exp_minus_iphil = np.exp(-_ci * phil_intermediate)
    exp_plus_iphil = np.exp(_ci * phil_intermediate)
    # Assign values to P matrices
    for wavelength_index in range(Nl):
        for layer_index in range(Nd - 2):
            P_matrices[wavelength_index, layer_index, 0, 0] = exp_minus_iphil[wavelength_index, layer_index]
            P_matrices[wavelength_index, layer_index, 1, 1] = exp_plus_iphil[wavelength_index, layer_index]
            P_matrices[wavelength_index, layer_index, 0, 1] = 0.0
            P_matrices[wavelength_index, layer_index, 1, 0] = 0.0
    return P_matrices



def _compute_pm(phil):
    """Compute the P matrices for each intermediate layer and wavelength"""
    _pm = np.eye(2, dtype=np.complex128)  # Identity matrix to avoid unnecessary zeros
    _ci = 1j  # Directly use imaginary unit

    _pm[0, 0] = np.exp(-_ci * phil)
    _pm[1, 1] = np.exp(_ci * phil)

    return _pm

#@jit(nopython=True)
def _compute_pm_analytical_gradient(kzl, phil):
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
    _pm_analytical_gradient = np.zeros((2, 2), dtype=np.complex128)
    _ci = 0 + 1j
    _a = -1 * _ci * phil
    _b = _ci * phil

    _pm_analytical_gradient[0, 0] = -_ci * kzl * np.exp(_a)
    _pm_analytical_gradient[1, 1] = _ci * kzl * np.exp(_b)

    return _pm_analytical_gradient


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


    def parse_input(self, args: dict) -> None:
        """
        Parse user inputs from a dictionary and set attributes for simulation.
        
        Parameters
        ----------
        args : dict
            Dictionary containing user input parameters.
        
        Raises
        ------
        ValueError
            If critical inputs are malformed or inconsistent, e.g.,
            mismatched lengths of thickness and material arrays.
        
        Returns
        -------
        None
        """


        # --- Incident Angles (list of degrees -> radians) ---
        incident_angles = args.get("incident_angle", [0.0])
        
        # If a scalar is provided instead of list, convert to list for uniformity
        if isinstance(incident_angles, (int, float)):
            incident_angles = [incident_angles]
        elif not isinstance(incident_angles, (list, tuple, np.ndarray)):
            raise ValueError(f"incident_angle must be a scalar or list/array, got {type(incident_angles)}")
        
        self.incident_angle = np.deg2rad(np.array(incident_angles, dtype=float))
        
        # --- Number of Angles ---
        self.number_of_angles = len(self.incident_angle)
        
        # --- Polarization ---
        # Default logic:
        # - If single angle at 0°, default to "p"
        # - Otherwise compute both "p" and "s" unless user explicitly provides a polarization list/string
        
        user_polarization = args.get("polarization", None)
        
        if self.number_of_angles == 1 and np.isclose(self.incident_angle[0], 0.0):
            # Single normal incidence angle: p and s are identical
            if user_polarization is None:
                self.polarization = ["p"]  # default to p polarization
            else:
                # accept single string or list with one polarization
                if isinstance(user_polarization, str):
                    self.polarization = [user_polarization.lower()]
                elif isinstance(user_polarization, (list, tuple)):
                    if len(user_polarization) != 1:
                        raise ValueError("polarization list length must be 1 for single incident angle")
                    self.polarization = [user_polarization[0].lower()]
                else:
                    raise ValueError("polarization must be string or list of strings")
        else:
            # Multiple angles or nonzero angle: polarization must cover both 'p' and 's' unless specified
            if user_polarization is None:
                self.polarization = ["p", "s"]
            else:
                # User provided polarization — accept string or list, normalize to list
                if isinstance(user_polarization, str):
                    self.polarization = [user_polarization.lower()]
                elif isinstance(user_polarization, (list, tuple)):
                    self.polarization = [p.lower() for p in user_polarization]
                else:
                    raise ValueError("polarization must be string or list of strings")

        # You can add validation here to ensure polarization values are valid ('p' and/or 's')
        valid_pols = {"p", "s"}
        if not all(p in valid_pols for p in self.polarization):
            raise ValueError(f"polarization entries must be 'p' or 's', got {self.polarization}")

        # --- Now self.incident_angle is an array of radians, 
        # --- self.number_of_angles matches its length,
        # --- and self.polarization is a list like ['p'] or ['p','s'] ready for downstream use.

        # --- Wavelength list and related arrays ---
        lamlist = args.get("wavelength_list", [400e-9, 800e-9, 10])
        if (
            not isinstance(lamlist, (list, tuple)) or
            len(lamlist) != 3 or
            not all(isinstance(x, (int, float)) for x in lamlist)
        ):
            raise ValueError(f"Invalid wavelength_list format: {lamlist}")
        
        start, stop, num = lamlist
        num = int(num)
        if num <= 0:
            raise ValueError(f"Number of wavelengths must be positive, got {num}")
        self.wavelength_array = np.linspace(start, stop, num)
        self.number_of_wavelengths = num
        self.wavenumber_array = 1 / self.wavelength_array

        # --- Thickness array ---
        thickness_list = args.get("thickness_list")
        if thickness_list is None:
            logger.warning("Thickness array not specified, proceeding with default [0, 900e-9, 0].")
            self.thickness_array = np.array([0, 900e-9, 0])
        else:
            self.thickness_array = np.array(thickness_list, dtype=float)

        # --- Material array ---
        material_list = args.get("material_list")
        if material_list is None:
            logger.warning("Material array not specified, proceeding with default ['Air', 'SiO2', 'Air'].")
            self.material_array = ["Air", "SiO2", "Air"]
        else:
            if not isinstance(material_list, (list, tuple)):
                raise ValueError("material_list must be a list or tuple")
            self.material_array = material_list
        
        self.number_of_layers = len(self.material_array)

        # --- Check consistency of thickness and material arrays ---
        if len(self.thickness_array) != self.number_of_layers:
            raise ValueError(
                f"Thickness array length ({len(self.thickness_array)}) does not match material array length ({self.number_of_layers})"
            )

        # --- Random thickness layers ---
        if "random_thickness_layers" in args:
            self.random_thickness_list = np.array(args["random_thickness_layers"], dtype=int)
        else:
            # Default: all layers except first and last (interface layers)
            if self.number_of_layers > 2:
                self.random_thickness_list = np.arange(1, self.number_of_layers - 1, dtype=int)
            else:
                self.random_thickness_list = np.array([], dtype=int)

        # --- Random thickness bounds in nanometers ---
        bounds = args.get("random_thickness_bounds_nm", [1, 1000])
        if (
            not isinstance(bounds, (list, tuple)) or
            len(bounds) != 2 or
            not all(isinstance(b, (int, float)) for b in bounds)
        ):
            raise ValueError(f"Invalid random_thickness_bounds_nm format: {bounds}")
        self.minimum_thickness_nm, self.maximum_thickness_nm = bounds

        # --- Random material layers ---
        if "random_material_layers" in args:
            self.random_materials_list = np.array(args["random_material_layers"], dtype=int)
        else:
            if self.number_of_layers > 2:
                self.random_materials_list = np.arange(1, self.number_of_layers - 1, dtype=int)
            else:
                self.random_materials_list = np.array([], dtype=int)

        # --- Possible random materials ---
        self.possible_materials = args.get(
            "possible_random_materials",
            ["SiO2", "Al2O3", "TiO2", "Ag", "Au", "Ta2O5"]
        )

        # --- Efficiency Weights with normalization ---
        def get_weight(key, default, name):
            value = args.get(key, default)
            logger.info(f"Using {name}: {value}")
            return float(value)

        tew = get_weight("transmission_efficiency_weight", 1.0/3.0, "Transmission Efficiency Weight (TEW)")
        rew = get_weight("reflection_efficiency_weight", 1.0/3.0, "Reflection Efficiency Weight (REW)")
        rsw = get_weight("reflection_selectivity_weight", 1.0/3.0, "Reflection Selectivity Weight (RSW)")

        total_weight = tew + rew + rsw
        if total_weight == 0:
            raise ValueError("Sum of efficiency weights cannot be zero.")
        self.transmission_efficiency_weight = tew / total_weight
        self.reflection_efficiency_weight = rew / total_weight
        self.reflection_selectivity_weight = rsw / total_weight

        # --- Gradient list ---
        if "gradient_list" in args:
            self.gradient_list = np.array(args["gradient_list"], dtype=int)
        else:
            if self.number_of_layers > 2:
                self.gradient_list = np.arange(1, self.number_of_layers - 1, dtype=int)
            else:
                self.gradient_list = np.array([], dtype=int)


        # --- Transmissive window (nm) ---
        transmissive_window_nm = args.get("transmissive_window_nm", [350, 700])
        if (
            not isinstance(transmissive_window_nm, (list, tuple)) or
            len(transmissive_window_nm) != 2
        ):
            raise ValueError(f"Invalid transmissive_window_nm format: {transmissive_window_nm}")

        self.transmissive_window_start_nm, self.transmissive_window_stop_nm = transmissive_window_nm
        self.transmissive_window_start = self.transmissive_window_start_nm * 1e-9
        self.transmissive_window_stop = self.transmissive_window_stop_nm * 1e-9

        self.transmissive_envelope = np.where(
            (self.wavelength_array >= self.transmissive_window_start) &
            (self.wavelength_array <= self.transmissive_window_stop),
            1.0,
            0.0
        )

        # --- Reflective window (wavenumbers cm^-1) ---
        reflective_window_wn = args.get("reflective_window_wn", [2000, 2400])
        if (
            not isinstance(reflective_window_wn, (list, tuple)) or
            len(reflective_window_wn) != 2
        ):
            raise ValueError(f"Invalid reflective_window_wn format: {reflective_window_wn}")

        self.reflective_window_start_wn, self.reflective_window_stop_wn = reflective_window_wn

        # Convert wavenumbers to wavelength in meters: λ = 1 / (wn in m^-1)
        # wavenumber is in cm^-1 so convert to m^-1 by multiplying by 100
        self.reflective_window_start = 1 / (self.reflective_window_stop_wn * 100)
        self.reflective_window_stop = 1 / (self.reflective_window_start_wn * 100)

        self.reflective_envelope = np.where(
            (self.wavelength_array >= self.reflective_window_start) &
            (self.wavelength_array <= self.reflective_window_stop),
            1.0,
            0.0
        )

        # --- PSC Thickness Option ---
        self.psc_thickness_option = int(args.get("psc_thickness_option", 200))

        # --- PV Lambda Bandgap ---
        self.pv_lambda_bandgap = args.get("pv_lambda_bandgap", 750e-9)

        # --- Load solar spectrum and atmospheric transmissivity ---
        self._solar_spectrum = self._read_AM()
        self._atmospheric_transmissivity = self._read_Atmospheric_Transmissivity()

    def set_refractive_index_array(self) -> None:
        """
        Define the refractive index array for each layer and wavelength based on specified materials.

        - Initializes refractive index array with dummy values.
        - Sets terminal layers to Air by default.
        - Assigns refractive indices to intermediate layers based on material names.
        - If material name is not recognized, attempts to load from file.

        Raises
        ------
        ValueError
            If the length of self.material_array does not match self.number_of_layers.
        """
        
        if len(self.material_array) != self.number_of_layers:
            raise ValueError(
                f"material_array length ({len(self.material_array)}) does not match number_of_layers ({self.number_of_layers})"
            )

        # Initialize refractive index array with complex ones
        ri_list = np.ones(self.number_of_layers, dtype=complex)
        self._refractive_index_array = np.tile(ri_list, (self.number_of_wavelengths, 1))

        # Default terminal layers to Air
        self.material_Air(0)
        self.material_Air(self.number_of_layers - 1)

        # Mapping from lowercase material string to corresponding method name
        material_methods = {
            "air": self.material_Air,
            "ag": self.material_Ag,
            "al": self.material_Al,
            "al2o3": self.material_Al2O3,
            "al2o3_udm": self.material_Al2O3_UDM,
            "aln": self.material_AlN,
            "au": self.material_Au,
            "hfo2": self.material_HfO2,
            "hfo2_udm": self.material_HfO2_UDM,  # from http://newad.physics.muni.cz/table-udm/HfO2-X2194-AO54_9108.Enk
            "hfo2_udm_no_loss": self.material_HfO2_UDM_v2,  # k set to 0.0
            "mgf2": self.material_MgF2_UDM,  # from http://newad.physics.muni.cz/table-udm/MgF2-X2935-SPIE9628.Enk
            "pb": self.material_Pb,
            "polystyrene": self.material_polystyrene,
            "pt": self.material_Pt,
            "re": self.material_Re,
            "rh": self.material_Rh,
            "ru": self.material_Ru,
            "si": self.material_Si,
            "sio2": self.material_SiO2,
            "sio2_udm": self.material_SiO2_UDM,
            "sio2_udm_v2": self.material_SiO2_UDM_v2,  # from http://newad.physics.muni.cz/table-udm/LithosilQ2-SPIE9890.Enk
            "ta2o5": self.material_Ta2O5,
            "tin": self.material_TiN,
            "tio2": self.material_TiO2,
            "w": self.material_W,
            "zro2": self.material_ZrO2,
            "si3n4": self.material_Si3N4,
        }

        # Assign refractive index for each internal layer
        for i in range(1, self.number_of_layers - 1):
            material_name = self.material_array[i]
            material_key = material_name.lower()

            if material_key in material_methods:
                logger.debug(f"Assigning refractive index for layer {i}: {material_name}")
                material_methods[material_key](i)
            else:
                # Assume material_name is a filename if not matched
                logger.info(f"Material '{material_name}' not recognized; loading from file for layer {i}.")
                self.material_from_file(i, material_name)

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


    def randomize_thickness_array(self):
        """Function to randomize the thickness array"""
        N = len(self.random_thickness_list)
        for i in range(N):
            _idx = self.random_thickness_list[i]
            _d = (
                np.random.randint(self.minimum_thickness_nm, self.maximum_thickness_nm)
                * 1e-9
            )
            self.thickness_array[_idx] = _d

    def randomize_materials_array(self):
        """Function to randomize the materials array"""
        # randomize the materials array
        N = len(self.random_materials_list)
        M = len(self.possible_materials)
        self.materials_code = np.zeros(N)
        for i in range(N):
            _idx = self.random_materials_list[i]
            _jdx = np.random.randint(0, M)
            self.material_array[_idx] = self.possible_materials[_jdx]
            self.materials_code[i] = _jdx

        # reset the refractive index array
        self.set_refractive_index_array()

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
        self._compute_phil()

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

    def compute_pv_stpv(self):
        """
        A method to compute the different figures of merit for PV-STPV, which 
        should follow a similar pattern as compute_stpv() on line 944
        
        Returns:
        --------
        None
        """

        # first compute or set temperature
        self.compute_self_consistent_temperature()

        # second compute short circuit current
        self.compute_pv_stpv_short_circuit_current()

        # third compute splitting power
        self.compute_pv_stpv_splitting_power()
        # probably JSCself._compute_pv_stpv_power_density(self.wavelength_array)

    def compute_pv_stpv_short_circuit_current(self):
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
        # scale AM by \lambda / \lambda_bg
        env[:bg_idx] = self.wavelength_array[:bg_idx] / self.pv_lambda_bandgap
        # compute the useful power density spectrum
        power_density_array = (
            self._solar_spectrum * absorptivity_full_stack * env
        )

        self.pv_stpv_short_circuit_current = np.trapz(
            power_density_array, self.wavelength_array
        )

        # go back to original spectrum
        self.remove_layer(_ln)
        self.compute_spectrum()
        emissivity_1_B = self.emissivity_array
        # get Blackbody spectrum at the default temperature - this is tentative
        self._compute_therml_spectrum()

        # now add perovskite layer to the stack and get the emissivity/absorptivity towards the sky
        self.reverse_stack()


        


    def compute_pv_stpv_short_circuit_current_gradient_gradient(self):
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


        _ln = len(self.thickness_array) - 1
        # insert thick active layer as the bottom-most layer
        self.insert_layer(_ln, 1000e-9)
        # make sure the active layer has RI of 2D perovskite
        self.material_2D_HOIP(_ln)
        # Acquire necessary variables
        self._solar_spectrum = self._read_AM()
        self.compute_spectrum()
        absorptivity_2_T = self.emissivity_array

        # get the absorbed power
        P_abs = np.trapz(absorptivity_2_T * self._solar_spectrum, self.wavelength_array)

        # loop over temperature to try to find the temperature of the stack that balances emitted
        # power with absorbed power
        _kill = 1
        while _kill:
            _T = 300
            _bbs = self._compute_blackbody_spectrum(self.wavelength_array, _T)
            P_emit = np.trapz(
                np.pi / 2 * _bbs * (emissivity_1_B + emissivity_1_T),
                self.wavelength_array,
            )
            _T += 1
            if P_emit > P_abs:
                _kill = 0

        self._compute_pv_stpv_power_density(self.wavelength_array)
        # reverse stack again and add active layer and get absorbed power into the structure
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

        # go back to original spectrum
        self.remove_layer(_ln)
        self.compute_spectrum()
        self.compute_spectrum_gradient()
    # Other figure of merit calculations here to be called in compute_pv_stpv

    def compute_pv_stpv_total_incident_power(self):
        """Docstring
        Use equation npv = Jsc * Voc * FF
        Jsc = short circuit current
        Voc = open circuit current
        FF = fill factor/ratio of ontainable power to short circuit * open circuit voltage
        
        The plan:

        initialze npv (assuming a static number), calculate Voc and FF, and multiply these three together to get total incident power as a unitless efficiency 
        Voc = (kB*Temperature/charge)*(ln(short circuit current)/(initial current))
        total_incident_power = pv_stpv_short_circuit_current_gradient * Voc * Fill factor
        
        """
        pass 
    
    def compute_pv_stpv_splitting_power_spectrum(self):
        """  
        Docstring

        Method to compute the pv_stpv splitting power spectrum as defined by 
         the integrand of Eq. (46) of https://www.overleaf.com/project/648a0cfeae29e31e10afc075 
        """

        # reverse the stack
        self.reverse_stack()
        # update emissivity
        self.compute_spectrum()
               
    
        # Store thermal emission spectra into the active layer
        self.pv_stpv_splitting_power_spectrum = self.blackbody_spectrum * self.emissivity_array

        # reverse the stack back
        self.reverse_stack()
        # update the spectra for the normal direction
        self.compute_spectrum()

        # approximate ideal spectral response assuming \lambda_bg = 700 nm
        self.lambda_bandgap = 700e-9
        self.spectral_response = self.wavelength_array / self.lambda_bandgap
        # make sure we have the solar spectrum
        self._solar_spectrum = self._read_AM()
        # now compute pv_stpv short circuit current
        self._compute_pv_short_circuit_current(
            self.wavelength_array,
            self.emissivity_array,
            self.spectral_response,
            self._solar_spectrum,
        )


    def compute_pv_stpv_splitting_power(self):
        
        """  
        Docstring

        Method to compute the pv_stpv splitting power as defined in Eq. (46) of https://www.overleaf.com/project/648a0cfeae29e31e10afc075 
        Attributes
        -----------
        emissivity_AR_polystyrene : array
                                    Storage of emissivity array.
        sliced_wavelength_array : array
                                    Slice of the wavelength array over the upper and lower limits determined by the minimum and maximum
                                    differences of array values and the integral's bounds.
        sliced_emissivity_array : array
                                    Slice of the emissivity array over the upper and lower limits determined by the minimum and maximum
                                    differences of array values and the integral's bounds.
        pv_stpv_splitting_power : array
                                    Emissivity array integrated over 3.0 to 3.5 microns.
        
        Returns
        -------
        None

        Notes:  The emissivity needs to be computed for the reversed original stack (meaning the stack *without the active layer*) before updating the 
                thermal emission spectrum.
                Steps:
                1. Reverse the stack
                2. Compute the optical spectra
                3. Compute the thermal emission spectra
                4. Define the integrand in Eq. (46)
                5. Integrate the integrand and store to the attribute self.pv_stpv_splitting_power
        
        """
               
        # Reverse stack, active layer was removed in the last function
        # Compute the optical and thermal spectra
        self.compute_pv_stpv_splitting_power_spectrum()

        # Set integration limits to lambda 1 to lambda 2 (3 um and 3.5 um) and store
        x_lower_limit = 3e-6
        x_upper_limit = 3.5e-6

        # Subtracts the limit from the original arrays, apply absolute value, and finds the minimum or maximum point
        wavelength_array_lower = np.abs(self.wavelength_array - x_lower_limit).argmin()
        wavelength_array_upper = np.abs(self.wavelength_array - x_upper_limit).argmin()
        
        sliced_wavelength_array = self.wavelength_array[wavelength_array_lower:wavelength_array_upper]
        sliced_splitting_spectrum = self.pv_stpv_splitting_power_spectrum[wavelength_array_lower:wavelength_array_upper]

        # Integrate over these slices and store
        self.pv_stpv_splitting_power = np.pi * np.trapz(sliced_splitting_spectrum, sliced_wavelength_array)


    def compute_self_consistent_temperature(self):
        """
        Method to compute the self-consistent temperature that balances emitted with absorbed power
        for pv-stpv applications
        """
        # this loops over temperatures starting with 300 K and stops when emitted power exceeds
        # absorbed power.
        # TO DO:
        # need to make sure we are computing the correct values of emissivity using the notes above
        _kill = 1
        _T = 300

        #while(_kill):
        #    _bbs = self._compute_blackbody_spectrum(self.wavelength_array, _T)
        #    P_emit = np.trapz( np.pi/2 * _bbs * (emissivity_A_B + absorptivity_AB_T), self.wavelength_array)
        #    _T += 1
        #    if P_emit > absorptivity_B_T :
        #        _kill = 0
        # assign _T to self.temperature
        self.temperature = _T


        
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

    def _compute_costheta(self):

        self._theta = np.zeros((self.number_of_wavelengths, self.number_of_layers), dtype=np.complex128)
        self._ctheta = np.zeros((self.number_of_wavelengths, self.number_of_layers), dtype=np.complex128)
        _CTHETA = np.zeros(num_layers, dtype=np.complex128)

        # Compute refraction angles
        _THETA[0] = self.incident_angle
        _CTHETA[0] = np.cos(self.incident_angle)
        _CTHETA[1:] = _kz[1:] / (_refractive_index[1:] * _k0)
        _THETA[1:] = np.arccos(_CTHETA[1:])

    def _compute_phil(self):
        """computes the phil angle for each layer and wavelength
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
        # compute phil_array using broadcasting
        self._phil_array = self._kz_array * self.thickness_array

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
        self._k0_array = np.pi * 2 / self.wavelength_array # in units of 1/m

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
        incident_angle = self.incident_angle.reshape(1,-1) # (1, N_angles)
        k0 = self._k0_array.reshape(-1, 1) # (N_wavelengths, 1)
        n0 = self._refractive_index_array[:, 0].reshape(-1, 1) # (N_wavelengths, 1)

        self._kx_array = ( n0 * np.sin(incident_angle) * k0 )

        ## compute kx_array
        #self._kx_array = (
        #    self._refractive_index_array[:, 0]
        #    * np.sin(self.incident_angle)
        #    * self._k0_array
        #)

    #def _compute_tm_gradient(self, _refractive_index, _k0, _kz, _d, _ln):
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

        return _tm_gradient, _THETA, _CTHETA """
    
    def _compute_tm_gradient(self, _refractive_index, _k0, _kz, _d, _ln):
        """Compute the transfer matrix gradient with respect to a layer _ln"""
        num_layers = self.number_of_layers

        _PHIL = _kz * _d
        print("PRINTING PHI_L EXPLICIT")
        print(_PHIL)
        print("PRINTING FROM PHI ARRAY")
        print(self._phil_array[0, :])

        _THETA = np.zeros(num_layers, dtype=np.complex128)
        _CTHETA = np.zeros(num_layers, dtype=np.complex128)

        # Compute refraction angles
        _THETA[0] = self.incident_angle
        _CTHETA[0] = np.cos(self.incident_angle)
        _CTHETA[1:] = _kz[1:] / (_refractive_index[1:] * _k0)
        _THETA[1:] = np.arccos(_CTHETA[1:])

        # Initialize transfer matrix
        _DM, _tm_gradient = _compute_dm(_refractive_index[0], _CTHETA[0], self.polarization)

        # Loop through layers
        for i in range(1, num_layers - 1):
            _DM, _DIM = _compute_dm(_refractive_index[i], _CTHETA[i], self.polarization)

            if i == _ln:
                _PM = _compute_pm_analytical_gradient(_kz[i], _PHIL[i])
            else:
                _PM = _compute_pm(_PHIL[i])

            # Use BLAS-optimized multiplications
            _tm_gradient = zgemm(1.0, _tm_gradient, _DM)
            _tm_gradient = zgemm(1.0, _tm_gradient, _PM)
            _tm_gradient = zgemm(1.0, _tm_gradient, _DIM)

        # Compute last layer contribution
        _DM, _DIM = _compute_dm(_refractive_index[-1], _CTHETA[-1], self.polarization)
        _tm_gradient = zgemm(1.0, _tm_gradient, _DM)

        return _tm_gradient, _THETA, _CTHETA


    def _compute_tm(self, _refractive_index, _k0, _kz, _d):
        """Compute the transfer matrix for each wavelength"""
        _PHIL = _kz * _d
        _THETA = np.zeros(self.number_of_layers, dtype=complex)
        _CTHETA = np.zeros(self.number_of_layers, dtype=complex)

        print("PRINTING PHI_L EXPLICIT")
        print(_PHIL)
        print("PRINTING FROM PHI ARRAY")
        print(self._phil_array[1, :])

        # Compute refraction angles
        _THETA[0] = self.incident_angle
        _CTHETA[0] = np.cos(self.incident_angle)
        _CTHETA[1:] = _kz[1:] / (_refractive_index[1:] * _k0)
        _THETA[1:] = np.arccos(_CTHETA[1:])

        # Initialize matrices
        _DM, _tm = _compute_dm(_refractive_index[0], _CTHETA[0], self.polarization)

        # Loop through layers (optimized)
        for i in range(1, self.number_of_layers - 1):
            _DM, _DIM = _compute_dm(_refractive_index[i], _CTHETA[i], self.polarization)
            _PM = _compute_pm(_PHIL[i])

            # Use BLAS-optimized multiplication
            _tm = zgemm(1.0, _tm, _DM)  # In-place multiplication
            _tm = zgemm(1.0, _tm, _PM)
            _tm = zgemm(1.0, _tm, _DIM)

        # Last layer computation
        _DM, _DIM = _compute_dm(_refractive_index[-1], _CTHETA[-1], self.polarization)
        _tm = zgemm(1.0, _tm, _DM)

        return _tm, _THETA, _CTHETA 
    
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

    def compute_selective_mirror_fom(self):
        """compute the figure of merit for selective tranmission and reflection according
           to the transmissive_envelope and reflective_envelope functions
        
        Attributes
        ----------
        self.transmissive_envelope : 1 x number_of_wavelength numpy array of floats 
            the box funcion that spans the window where we want high transmissivity

        self.reflective_envelope : 1 x number_of_wavelength numpy array of floats 
            the box function that spans the window where we want high reflectivity

        self.reflection_efficiency : float
            the first reflection fom: (int R(lambda) * reflective_envelope d lambda) / (int R(lambda) d lambda)

        self.reflection_selectivity : float
            the second reflection fom: (int R(lambda) * reflective_envelope d lambda) / (int reflective_envelope d lambda) 

        self.transmission_efficiency : float
            the transmission fom: (int T(lambda) * transmissive_envelope d lambda) / (int tranmissive_envelope d lambda)

        self.selective_mirror_fom : float
            the composite figure of merit defined as f = a * transmission_efficiency + b * reflection_efficiency + c * reflection_selectivity

        self.transmission_efficiency_weight : float
            the weight of transmission efficiency in the composite figure of merit (a) satisfying a + b + c = 1

        self.reflection_efficiency_weight : float
            the weight of the reflection efficiency in the composite figure of merit (b) satisfying a + b + c = 1

        self.reflection_selectivity_weight : float
            the weight of the reflection selectivity in the composite figure of merit (c) satisfying a + b + c = 1
        
        Returns
        -------
        None

        """
        # numerators come from the actual spectra times the envelope
        _ut_array = self.transmissive_envelope * self.transmissivity_array
        _ur_array = self.reflective_envelope * self.reflectivity_array

        # integrate to get numerators
        _ut = np.trapz(_ut_array, self.wavelength_array)
        _ur = np.trapz(_ur_array, self.wavelength_array)

        # denominators are slightly different between R and T.

        # T_denom -> integrate transmissive envelope
        _t_denom = np.trapz(
            self.transmissive_envelope, self.wavelength_array
        )

        # R_denom -> integrate reflection spectrum
        _r_denom = np.trapz(
            self.reflectivity_array, self.wavelength_array
        )

        # R_selective_denom -> integrate the reflection envelope
        _r_select_denom = np.trapz(
           self.reflective_envelope, self.wavelength_array 
        )

        # if transmissivity_envelope is zero everywhere, this will give nan.. handle
        # by just giving value of zero the transmission_efficiency

        if _t_denom == 0.0:
            self.transmission_efficiency = 0.0
        else:
            self.transmission_efficiency = (_ut / _t_denom)

        # if reflectivity is zero everywhere, this will give nan - handle
        # by just giving value of zero to reflection_efficiency
        
        if _r_denom == 0.0:
            self.reflection_efficiency = 0.0
        else: 
            self.reflection_efficiency = (_ur / _r_denom)

        if _r_select_denom == 0.0:
            self.reflection_selectivity = 0.0
        else:
            self.reflection_selectivity = (_ur / _r_select_denom) 

        self.selective_mirror_fom = (
            self.transmission_efficiency_weight * self.transmission_efficiency
            + self.reflection_efficiency_weight * self.reflection_efficiency
            + self.reflection_selectivity_weight * self.reflection_selectivity
        )

    def compute_selective_mirror_fom_gradient(self):
        """compute the figure of merit for selective tranmission and reflection according
           to the transmissive_envelope and reflective_envelope functions
        
        Attributes
        ----------
        self.transmissive_envelope : 1 x number_of_wavelength numpy array of floats 
            the box funcion that spans the window where we want high transmissivity

        self.reflective_envelope : 1 x number_of_wavelength numpy array of floats 
            the box function that spans the window where we want high reflectivity

        self.reflection_efficiency : float
            the reflection fom: (int R(lambda) * reflective_envelope d lambda) / (int R(lambda) d lambda)

        self.transmission_efficiency : float
            the transmission fom: (int T(lambda) * transmissive_envelope d lambda) / (int tranmissive_envelope d lambda)

        self.selective_mirror_fom : float
            the composite figure of merit defined as f = a * transmission_efficiency + b * reflection_efficiency

        self.transmission_efficiency_weight : float
            the weight of transmission efficiency in the composite figure of merit (a) satisfying a + b = 1

        self.reflection_efficiency_weight : float
            the weight of the reflection efficiency in the composite figure of merit (b) satisfying a + b = 1
        
        Returns
        -------
        None
        
        
        compute the figure of merit for selective tranmission and reflection according
        to the transmissive_envelope and reflective_envelope functions

        Working Equation
        ----------------
        useful_transmitted_power = int T(lambda) * transmission_envelope d lambda

        transmission_denom = int transmission_envelope( lambda) d lambda

        useful_reflected_power = int R(lambda) * reflection_envelope d lambda

        total_reflected_power = int R(lambda) d lambda

        eta_T' = int T'(lambda) * transmission_envelope d lambda / transmission_denom

        eta_R' = g(lambda) f'(lambda) - f(lambda) g'(lambda)  / g(lambda) ^ 2

        where g(lambda) = int R(lambda) d lambda
              f(lambda) = int reflectivity_envelope R(lambda) d lambda
              g'(lambda) = int R'(lambda) dlambda
              f'(lambda) = int reflectivity_envelope R'(lambda) d lambda

        """
        # eta_T' = Pi(lambda) * T'(lambda) / Pi(lambda)
        self.compute_spectrum_gradient()

        _ngr = len(self.transmissivity_gradient_array[0, :])
        # integrate the thermal emission spectrum over wavelength using np.trapz
        self.transmission_efficiency_gradient = np.zeros(_ngr)
        self.reflection_efficiency_gradient = np.zeros(_ngr)
        self.reflection_selectivity_gradient = np.zeros(_ngr)

        # this term is in the denominator of each of the eta_T' elements
        _eta_T_denom = np.trapz(self.transmissive_envelope, self.wavelength_array)

        # this term is in the denominator of each of the sel_R' elements
        _sel_R_denom = np.trapz(self.reflective_envelope, self.wavelength_array)

        # these terms are in each of the eta_R' elements
        _f_l = np.trapz(
            self.reflective_envelope * self.reflectivity_array, self.wavelength_array
        )
        _g_l = np.trapz(self.reflectivity_array, self.wavelength_array)

        for i in range(_ngr):
            # can compute eta_T' in one shot
            self.transmission_efficiency_gradient[i] = (
                np.trapz(
                    self.transmissive_envelope
                    * self.transmissivity_gradient_array[:, i],
                    self.wavelength_array,
                )
                / _eta_T_denom
            )
            # can compute sel_R'
            self.reflection_selectivity_gradient[i] = (
                np.trapz(
                    self.reflective_envelope 
                    * self.reflectivity_gradient_array[:,i],
                    self.wavelength_array
                )
                / _sel_R_denom
            )

            # need to get parts of g'(lambda) and f'(lambda) terms
            _gp_l = np.trapz(
                self.reflectivity_gradient_array[:, i], self.wavelength_array
            )
            _fp_l = np.trapz(
                self.reflective_envelope * self.reflectivity_gradient_array[:, i],
                self.wavelength_array,
            )

            self.reflection_efficiency_gradient[i] = (
                _g_l * _fp_l - _f_l * _gp_l
            ) / _g_l**2

        self.selective_mirror_fom_gradient = (
            self.transmission_efficiency_weight * self.transmission_efficiency_gradient
            + self.reflection_efficiency_weight * self.reflection_efficiency_gradient
            + self.reflection_selectivity_weight * self.reflection_selectivity_gradient
        )
