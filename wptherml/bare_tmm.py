from .spectrum_driver import SpectrumDriver
from .materials import Materials
import numpy as np
from scipy.linalg.blas import zgemm  # BLAS-optimized complex matrix multiplication
import logging

logger = logging.getLogger(__name__)


# --- Helper function to compute D and D_inv matrices vectorized ---
def compute_dm_vectorized(n_layer, cos_t, pol):
    shape = n_layer.shape + (2, 2)
    D = np.zeros(shape, dtype=np.complex128)
    D_inv = np.zeros_like(D)

    if pol == "s":
        D[..., 0, 0] = 1
        D[..., 0, 1] = 1
        D[..., 1, 0] = n_layer * cos_t
        D[..., 1, 1] = -n_layer * cos_t
    else:  # p-polarization
        D[..., 0, 0] = cos_t
        D[..., 0, 1] = cos_t
        D[..., 1, 0] = n_layer
        D[..., 1, 1] = -n_layer

    det = D[..., 0, 0] * D[..., 1, 1] - D[..., 0, 1] * D[..., 1, 0]
    inv_det = 1 / det

    D_inv[..., 0, 0] = inv_det * D[..., 1, 1]
    D_inv[..., 0, 1] = -inv_det * D[..., 0, 1]
    D_inv[..., 1, 0] = -inv_det * D[..., 1, 0]
    D_inv[..., 1, 1] = inv_det * D[..., 0, 0]

    return D, D_inv

    # --- Helper function to compute P matrices vectorized ---
    def compute_pm_vectorized(phil_layer):
        # phil_layer shape: (N_lambda, N_theta)
        shape = phil_layer.shape + (2, 2)
        P = np.zeros(shape, dtype=np.complex128)
        exp_minus = np.exp(-1j * phil_layer)
        exp_plus = np.exp(1j * phil_layer)

        P[..., 0, 0] = exp_minus
        P[..., 1, 1] = exp_plus
        return P


def batch_matmul(A, B):
    """
    Batch multiply two arrays of matrices A and B.
    A, B shape: (..., 2, 2)
    Returns: (..., 2, 2)
    """
    return np.einsum('...ij,...jk->...ik', A, B)


class TmmDriver(SpectrumDriver, Materials):
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

    
    def compute_spectrum(self):
        """
        Compute reflectivity, transmissivity, and emissivity spectra over all
        wavelengths, incident angles, and polarizations.

        Results are stored as attributes:
            reflectivity_array : shape (N_lambda, N_theta, N_pol)
            transmissivity_array : shape (N_lambda, N_theta, N_pol)
            emissivity_array : shape (N_lambda, N_theta, N_pol)
        """

        # Ensure k-vectors are up to date for all angles & wavelengths
        self._compute_k0()  # (N_lambda,)
        self._compute_kx()  # Should handle vector incident_angle now (N_lambda, N_theta)
        self._compute_kz()  # Should compute (N_lambda, N_theta, N_layers)

        # Compute transfer matrices M for all lambda, theta, pol
        M = self._compute_tm()  # shape (N_lambda, N_theta, N_pol, 2, 2)

        N_lambda, N_theta, N_pol = M.shape[:3]
        N_layers = self.number_of_layers

        # Prepare arrays to hold R, T, E
        self.reflectivity_array = np.empty((N_lambda, N_theta, N_pol))
        self.transmissivity_array = np.empty((N_lambda, N_theta, N_pol))
        self.emissivity_array = np.empty((N_lambda, N_theta, N_pol))

        # Extract relevant refractive indices and cosines for incident and final layers
        n_incident = self._refractive_index_array[:, 0].real[:, np.newaxis]    # (N_lambda, 1)
        n_final = self._refractive_index_array[:, -1].real[:, np.newaxis]      # (N_lambda, 1)

        # Cos(theta) in incident layer and final layer (shape: N_lambda x N_theta)
        cos_theta_incident = np.cos(self.incident_angle)[np.newaxis, :]       # (1, N_theta)
        # For final layer cos(theta) = kz / (n * k0) -- shape (N_lambda, N_theta)
        kz_final = self._kz_array[:, :, -1]
        n_final_expanded = self._refractive_index_array[:, np.newaxis, -1]
        k0_expanded = self._k0_array[:, np.newaxis]
        cos_theta_final = kz_final / (n_final_expanded * k0_expanded)  # (N_lambda, N_theta)

        # Loop over polarizations to fill arrays
        for p_idx in range(N_pol):
            # r and t for all wavelengths and angles
            r = M[:, :, p_idx, 1, 0] / M[:, :, p_idx, 0, 0]   # (N_lambda, N_theta)
            t = 1.0 / M[:, :, p_idx, 0, 0]

            # Reflectivity R = |r|^2
            R = np.abs(r) ** 2

            # Transmission factor with refractive indices and cosines
            factor = (n_final * cos_theta_final) / (n_incident * cos_theta_incident)  # shape (N_lambda, N_theta)

            # Transmissivity T = |t|^2 * factor
            T = np.abs(t) ** 2 * factor

            # Emissivity E = 1 - R - T
            E = 1.0 - R - T

            # Store results
            self.reflectivity_array[:, :, p_idx] = R
            self.transmissivity_array[:, :, p_idx] = T
            self.emissivity_array[:, :, p_idx] = E


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
        # refractive_index_array shape (N_wavelengths, N_layers)
        # expand to (N_wavelengths, 1, N_layers)
        n = self._refractive_index_array[:, np.newaxis, :]  # broadcast over angles

        # _kx_array shape (N_wavelengths, N_angles), expand last dim
        kx = self._kx_array[:, :, np.newaxis]  # (N_wavelengths, N_angles, 1)

        # k0 shape (N_wavelengths, 1, 1) for broadcasting
        k0 = self._k0_array[:, np.newaxis, np.newaxis]

        # Compute kz (complex) with broadcasting
        self._kz_array = np.sqrt((n * k0) ** 2 - kx ** 2)  # (N_wavelengths, N_angles, N_layers)

        #self._kz_array = np.sqrt(
        #    (self._refractive_index_array * self._k0_array[:, np.newaxis]) ** 2
        #    - self._kx_array[:, np.newaxis] ** 2
        #)

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



    def _compute_tm(self):
        """
        Vectorized computation of transfer matrix M for all wavelengths, angles, and polarizations.

        Returns:
            M: np.ndarray, shape (N_lambda, N_theta, N_pol, 2, 2), complex transfer matrices
        """

        N_lambda = self.number_of_wavelengths
        N_theta = self.number_of_angles
        N_pol = len(self.polarization)
        N_layers = self.number_of_layers

        # --- Broadcast refractive index to (N_lambda, N_theta, N_layers) ---
        n = self._refractive_index_array[:, np.newaxis, :]  # (N_lambda, 1, N_layers)
        n = np.broadcast_to(n, (N_lambda, N_theta, N_layers))

        # --- Broadcast incident angles and k0 ---
        angles = self.incident_angle[np.newaxis, :]  # (1, N_theta)
        k0 = self._k0_array[:, np.newaxis]  # (N_lambda, 1)

        # --- Compute kx array (N_lambda, N_theta) ---
        n0 = self._refractive_index_array[:, 0].real[:, np.newaxis]  # (N_lambda, 1)
        kx = n0 * np.sin(angles) * k0  # (N_lambda, N_theta)

        # --- Compute kz array (N_lambda, N_theta, N_layers) ---
        kz = np.sqrt((n * k0[:, :, np.newaxis]) ** 2 - kx[:, :, np.newaxis] ** 2)

        # --- Compute cos(theta) ---
        cos_theta = np.empty_like(kz, dtype=np.complex128)
        cos_theta[:, :, 0] = np.cos(angles)  # incident layer cos(theta)
        cos_theta[:, :, 1:] = kz[:, :, 1:] / (n[:, :, 1:] * k0[:, :, np.newaxis])

        # --- Compute phase thickness phil (N_lambda, N_theta, N_layers) ---
        phil = kz * self.thickness_array[np.newaxis, np.newaxis, :]  # broadcast thickness

        # --- Initialize output M tensor ---
        M = np.empty((N_lambda, N_theta, N_pol, 2, 2), dtype=np.complex128)

        # --- Prepare identity matrices for initialization ---
        identity = np.eye(2, dtype=np.complex128)

        # --- Main loop over polarization ---
        for p_idx, pol in enumerate(self.polarization):
            # Compute D and D_inv for all layers
            D, D_inv = compute_dm_vectorized(n, cos_theta, pol)  # shapes (N_lambda, N_theta, N_layers, 2, 2)

            # Initialize M to identity for all (lambda, theta)
            M_pol = np.broadcast_to(identity, (N_lambda, N_theta, 2, 2)).copy()

            # Loop through layers:
            # Multiply: M = M @ D_layer @ P_layer @ D_inv_layer for layers 1 to N_layers-2
            # And first multiply by D_0 and last multiply by D_{N_layers-1}

            # First multiply by D_0
            M_pol = batch_matmul(M_pol, D[:, :, 0, :, :])

            for layer in range(1, N_layers - 1):
                P_layer = compute_pm_vectorized(phil[:, :, layer])  # (N_lambda, N_theta, 2, 2)
                M_pol = batch_matmul(M_pol, P_layer)
                M_pol = batch_matmul(M_pol, D_inv[:, :, layer, :, :])
                M_pol = batch_matmul(M_pol, D[:, :, layer, :, :])

            # Multiply by last layer D
            M_pol = batch_matmul(M_pol, D[:, :, -1, :, :])

            # Store result
            M[:, :, p_idx, :, :] = M_pol

        return M



    

import numpy as np
import logging

# Setup logging to show info messages
logging.basicConfig(level=logging.INFO)

# Example input dictionary
test_args = {
    "wavelength_list": [400e-9, 800e-9, 100],  # 100 wavelengths from 400 to 800 nm
    "material_list": ["Air", "SiO2", "Au", "Air"],
    "thickness_list": [0, 200e-9, 10e-9, 0],
    # You can specify incident angle as scalar or list, default is [0.0]
    # "incident_angle": [0.0],
    # You can specify polarization, but default works fine for single angle
    # "polarization": "p",
}

def run_tmm_example():
    # Assuming TmmDriver is imported and available
    driver = TmmDriver(test_args)

    # The constructor calls parse_input, set_refractive_index_array, and compute_spectrum already

    # Let's print the shapes of computed spectra
    print(f"Reflectivity array shape: {driver.reflectivity_array.shape}")   # Expected (100, 1, 1)
    print(f"Transmissivity array shape: {driver.transmissivity_array.shape}")
    print(f"Emissivity array shape: {driver.emissivity_array.shape}")

    # Print spectra at normal incidence and p-polarization (index 0)
    wavelengths_nm = driver.wavelength_array * 1e9

    R = driver.reflectivity_array[:, 0, 0]
    T = driver.transmissivity_array[:, 0, 0]
    E = driver.emissivity_array[:, 0, 0]

    print("\nWavelength (nm) | Reflectivity | Transmissivity | Emissivity")
    print("-" * 55)
    for wl, r, t, e in zip(wavelengths_nm, R, T, E):
        print(f"{wl:10.1f}       | {r:10.4f}    | {t:12.4f}   | {e:9.4f}")

if __name__ == "__main__":
    run_tmm_example()

