import numpy as np
from scipy.linalg.blas import zgemm  # BLAS-optimized complex matrix multiplication
import logging
from matplotlib import pyplot as plt
logger = logging.getLogger(__name__)
from .spectrum_driver import SpectrumDriver
from .materials import Materials

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

    #print("GOING TO PRINT D")
    #print(D)
    #print("GOING TO PRINT D_INV")
    #print(D_inv)

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
    #print("GOING TO PRINT P")
    #print(P)
    return P


def batch_matmul(A, B):
    """
    Batch multiply two arrays of matrices A and B.
    A, B shape: (..., 2, 2)
    Returns: (..., 2, 2)
    """
    return np.einsum('...ij,...jk->...ik', A, B)

class VecTmmDriver(SpectrumDriver, Materials):

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
            elif _lm == "al2o3_udm":
                self.material_Al2O3_UDM(i)
            elif _lm == "aln":
                self.material_AlN(i)
            elif _lm == "au":
                self.material_Au(i)
            elif _lm == "hfo2":
                self.material_HfO2(i)
            elif _lm == "hfo2_udm":
                self.material_HfO2_UDM(i) #<== from this source http://newad.physics.muni.cz/table-udm/HfO2-X2194-AO54_9108.Enk - recommended for ALD HfO2
            elif _lm == "hfo2_udm_no_loss":
                self.material_HfO2_UDM_v2(i) #<== from this source http://newad.physics.muni.cz/table-udm/HfO2-X2194-AO54_9108.Enk but k set to 0.0
            elif _lm == "mgf2":
                self.material_MgF2_UDM(i) #<== from this source http://newad.physics.muni.cz/table-udm/MgF2-X2935-SPIE9628.Enk - recommended for ALD MgF2
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
            elif _lm == "sio2_udm": 
                self.material_SiO2_UDM(i)
            elif _lm == "sio2_udm_v2": #<== from this source http://newad.physics.muni.cz/table-udm/LithosilQ2-SPIE9890.Enk - recommended for ALD SiO2
                self.material_SiO2_UDM_v2(i)
            elif _lm == "ta2o5":
              self.material_Ta2O5(i)
            elif _lm == "tin":
                self.material_TiN(i)
            elif _lm == "tio2":
                self.material_TiO2(i)
            elif _lm == "w":
                self.material_W(i)
            elif _lm == "zro2":
                self.material_ZrO2(i)
            elif _lm == "si3n4":
                self.material_Si3N4(i)
            # if we don't match one of these strings, then we assume the user has passed
            # a filename
            else:
                self.material_from_file(i, _original_string)

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

    def _compute_tm(self):
        N_lambda = self.number_of_wavelengths
        N_theta = self.number_of_angles
        N_pol = len(self.polarization)
        N_layers = self.number_of_layers

        # broadcast refractive index array to match angles and layers
        n = self._refractive_index_array[:, np.newaxis, :]  # (N_lambda, 1, N_layers)
        n = np.broadcast_to(n, (N_lambda, N_theta, N_layers))

        angles = self.incident_angle[np.newaxis, :]  # (1, N_theta)
        k0 = self._k0_array[:, np.newaxis]  # (N_lambda, 1)

        n0 = self._refractive_index_array[:, 0].real[:, np.newaxis]  # (N_lambda, 1)
        kx = n0 * np.sin(angles) * k0  # (N_lambda, N_theta)



        kz = np.sqrt((n * k0[:, :, np.newaxis]) ** 2 - kx[:, :, np.newaxis] ** 2) # (N_lambda, N_theta, 1)

        cos_theta = np.empty_like(kz, dtype=np.complex128) # (N_lambda, N_theta, 1)
        cos_theta[:, :, 0] = np.cos(angles)
        cos_theta[:, :, 1:] = kz[:, :, 1:] / (n[:, :, 1:] * k0[:, :, np.newaxis])


        phil = kz * self.thickness_array[np.newaxis, np.newaxis, :]
        P_tensor = compute_pm_vectorized(phil[:, :, :])


        M = np.empty((N_lambda, N_theta, N_pol, 2, 2), dtype=np.complex128)
        identity = np.eye(2, dtype=np.complex128)


        for p_idx, pol in enumerate(self.polarization):
            D, D_inv = compute_dm_vectorized(n, cos_theta, pol)

            M_pol = np.broadcast_to(identity, (N_lambda, N_theta, 2, 2)).copy()

            M_pol = batch_matmul(M_pol, D_inv[:, :, 0, :, :])

            # Loop layers 1..N_layers-2
            for layer in range(1, N_layers - 1):
                P_layer = P_tensor[:, :, layer] #compute_pm_vectorized(phil[:, :, layer])

                M_pol = batch_matmul(M_pol, D[:, :, layer, :, :])
                M_pol = batch_matmul(M_pol, P_layer)
                M_pol = batch_matmul(M_pol, D_inv[:, :, layer, :, :])



            M_pol = batch_matmul(M_pol, D[:, :, -1, :, :])

            M[:, :, p_idx, :, :] = M_pol

        return M

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
        #print(F"GOING TO PRINT M for all wavelengths with angle=0 and pol = 'p'")
        #print(M[:, 0, 0, :, :])

        N_lambda, N_theta, N_pol = M.shape[:3]
        N_layers = self.number_of_layers

        # Prepare arrays to hold R, T, E
        self.reflectivity_array_full = np.empty((N_lambda, N_theta, N_pol))
        self.transmissivity_array_full = np.empty((N_lambda, N_theta, N_pol))
        self.emissivity_array_full = np.empty((N_lambda, N_theta, N_pol))

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
            self.reflectivity_array_full[:, :, p_idx] = np.real(R)
            self.transmissivity_array_full[:, :, p_idx] = np.real(T)
            self.emissivity_array_full[:, :, p_idx] = np.real(E)

        # store the normal incidence values in the first angle
        self.reflectivity_array = self.reflectivity_array_full[:, 0, 0]
        self.transmissivity_array = self.transmissivity_array_full[:, 0, 0]
        self.emissivity_array = self.emissivity_array_full[:, 0, 0]


    
