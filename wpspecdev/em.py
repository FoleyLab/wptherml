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

        _refraction_angle_array : 1 x number_of_layers numpy array of complex floats
            the incident and refraction angles for each layer, including incoming layer

        _cos_of_refraction_angle_array : 1 x number_of_layers numpy array of complex floats

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

    def __init__(self, thickness):
        """constructor for the TmmDriver class

        Assign values for attributes thickness_array, material_array then call
        compute_spectrum to compute values for attributes reflectivity_array,
        transmissivity_array, and emissivity_array

        """

        """ hard-coded a lot of this for now, we will obviously generalize soon! """
        self.thickness = thickness
        self.number_of_wavelengths = 10
        self.number_of_layers = 3
        self.wavelength_array = np.linspace(400e-9, 800e-9, self.number_of_wavelengths)
        self.wavenumber_array = 1 / self.wavelength_array
        self.thickness_array = np.array([0, thickness, 0])
        self.polarization = "s"
        self.incident_angle = 0.0

        # pretty pythonic way to create the _refractive_index_array
        # that will result in self._refractive_index_array[1, 3] -> RI of layer index 1 (2nd layer)
        # at wavelength index 3 (4th wavelength in wavelength_array)
        self._refractive_index_array = np.reshape(
            np.tile(np.array([1 + 0j, 1.5 + 0j, 1 + 0j]), self.number_of_wavelengths),
            (self.number_of_wavelengths, self.number_of_layers),
        )

    def parse_input(self, args):
        if "wavelength_list" in args:
            lamlist = args["wavelength_list"]
            self.wavelength_array = np.linspace(lamlist[0], lamlist[1], int(lamlist[2]))
            self.number_of_wavelengths = int(lamlist[2])
        else:
            self.wavelength_array = np.linspace(400e-9, 800e-9, 10)
            self.number_of_wavelengths = 10

        if "thickness_list" in args:
            self.thickness_array = args["thickness_list"]
        ### default structure
        else:
            print("  Thickness array not specified!")
            print("  Proceeding with default structure - optically thick W! ")
            self.thickness_array = [0, 900e-9, 0]
            self.material_array = ["Air", "SiO2", "Air"]

        if "material_list" in args:
            self.material_array = args["material_list"]
        else:
            print("  Material array not specified!")
            print("  Proceeding with default structure - optically thick W! ")
            self.thickness_array = [0, 900e-9, 0]
            self.material_array = ["Air", "SiO2", "Air"]

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

        Will compute attributes by

            - calling _compute_tm method

            - evaluating r amplitudes from _tm

            - evaluationg R from rr*

            - evaluating t amplitudes from _tm

            - evaluating T from tt* n_L cos(\theta_L) / n_1 cos(\theta_L)

        """


        """ continute to compute remaining intermediate attributes needed by _compute_tm(), including
        
            - self._refraction_angle_array
            
            - self._cos_of_refraction_angle_array
        
            - self._kz_array
            
        """

        # with all of these formed, you can now call _compute_tm()
        self._tm = self._compute_tm()



    def _compute_kz(self):
        """ computes the z-component of the wavevector in each layer of the stack
        
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
            (ml._refractive_index_array * ml._k0_array[:, np.newaxis]) ** 2
            - ml._kx_array[:, np.newaxis] ** 2
        )

    def _compute_k0(self):
        """ computes the _k0_array
        
        Attributes
        ----------
            wavelength_array : 1 x number of wavelengths float
                the wavelengths that will illuminate the structure in SI units
            
            _k0_array : 1 x number of wavelengths float
                the wavenumbers that will illuminate the structure in SI units
        
        """
        self._k0_array = np.pi * 2 / self.wavelength_array

    def _compute_kx(self):
        """ computes the _kx_array
        
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

    def _compute_tm(self):
        """compute the transfer matrix for each wavelength

        Attributes
        ----------
            thickness_array

            _k0

            _kx

            _kz_array

            _refraction_angle_array

            _cos_of_refraction_angle_array

            _dm

            _pm

            _dim



        Returns
        -------
        _tm

        """

def tmm(k0, theta0, pol, nA, tA):
    t1 = np.zeros((2,2),dtype=complex)
    t2 = np.zeros((2,2),dtype=complex)
    D1 = np.zeros((2,2),dtype=complex)
    Dl = np.zeros((2,2),dtype=complex)
    Dli = np.zeros((2,2),dtype=complex)
    Pl = np.zeros((2,2),dtype=complex)
    M  = np.zeros((2,2),dtype=complex)
    L = len(nA)
    kz = np.zeros(L,dtype=complex)
    phil = np.zeros(L,dtype=complex)
    ctheta = np.zeros(L,dtype=complex)
    theta = np.zeros(L,dtype=complex)
    ctheta[0] = np.cos(theta0)
    
    D1 = BuildD(nA[0], ctheta[0], pol)
    ### Note it is actually faster to invert the 2x2 matrix
    ### "By Hand" than it is to use linalg.inv
    ### and this inv step seems to be the bottleneck for the TMM function
    tmp = D1[0,0]*D1[1,1]-D1[0,1]*D1[1,0]
    det = 1/tmp
    M[0,0] = det*D1[1,1]
    M[0,1] = -det*D1[0,1]
    M[1,0] = -det*D1[1,0]
    M[1,1] = det*D1[0,0]
    #D1i = inv(D1)
   #print("D1i is ")
   #print(D1i)
    
    
    ### This is the number of layers in the structure

    
    ### since kx is conserved through all layers, just compute it
    ### in the upper layer (layer 1), for which you already known
    ### the angle of incidence
    kx = nA[0]*k0*np.sin(theta0)
    kz[0] = np.sqrt((nA[0]*k0)**2 - kx**2)
    kz[L-1] = np.sqrt((nA[L-1]*k0)**2 - kx**2)
    
    ### keeping consistent with K-R excitation
    if np.real(kz[0])<0:
        kz[0] = -1*kz[0]
    if np.imag(kz[L-1])<0:
        kz[L-1] = -1*kz[L-1]
    ### loop through all layers 2 through L-1 and compute kz and cos(theta)...
    ### note that when i = 1, we are dealing with layer 2... when 
    ### i = L-2, we are dealing with layer L-1... this loop only goes through
    ### intermediate layers!
    for i in range(1,(L-1)):
        kz[i] = np.sqrt((nA[i]*k0)**2 - kx**2)
        if np.imag(kz[i])<0:
            kz[i] = -1*kz[i]
        
        ctheta[i] = kz[i]/(nA[i]*k0)
        theta[i] = np.arccos(ctheta[i])

        phil[i] = kz[i]*tA[i]

        Dl = BuildD(nA[i],ctheta[i], pol)
        ## Invert Dl
        tmp = Dl[0,0]*Dl[1,1]-Dl[0,1]*Dl[1,0]
        det = 1/tmp
        Dli[0,0] = det*Dl[1,1]
        Dli[0,1] = -det*Dl[0,1]
        Dli[1,0] = -det*Dl[1,0]
        Dli[1,1] = det*Dl[0,0]
        #Dli = inv(Dl)
        ## form Pl
        Pl = BuildP(phil[i])

        t1 = np.matmul(M,Dl)
        t2 = np.matmul(t1,Pl)
        M  = np.matmul(t2,Dli)
        
    ### M is now the product of D_1^-1 .... D_l-1^-1... just need to 
    ### compute D_L and multiply M*D_L
    kz[L-1] = np.sqrt((nA[L-1]*k0)**2 - kx**2)
    ctheta[L-1]= kz[L-1]/(nA[L-1]*k0)
    DL = BuildD(nA[L-1], ctheta[L-1], pol)
    t1 = np.matmul(M,DL)
    ### going to create a dictionary called M which will 
    ### contain the matrix elements of M as well as 
    ### other important quantities like incoming and outgoing angles
    theta[0] = theta0
    theta[L-1] = np.arccos(ctheta[L-1])
    ctheta[0] = np.cos(theta0)
    M = {"M11": t1[0,0], 
         "M12": t1[0,1], 
         "M21": t1[1,0], 
         "M22": t1[1,1],
         "theta_i": theta0,
         "theta_L": np.real(np.arccos(ctheta[L-1])),
         "kz": kz,
         "phil": phil,
         "ctheta": ctheta,
         "theta": theta
         }

    return M

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

        _dm = np.zeros((2,2),dtype=complex)
        _dim = np.zeros((2,2),dtype=complex)

        if self.polarization=="s":
            _dm[0,0] = 1+0j
            _dm[0,1] = 1+0j
            _dm[1,0] = refractive_index * cosine_theta
            _dm[1,1] = -1 * refractive_index * cosine_theta

        elif self.polarization=="p":
            _dm[0,0] = cosine_theta+0j
            _dm[0,1] = cosine_theta+0j
            _dm[1,0] = refractive_index
            _dm[1,1] = -1 * refractive_index

        # Note it is actually faster to invert the 2x2 matrix
        # "By Hand" than it is to use linalg.inv
        # and this inv step seems to be the bottleneck for the TMM function
        # but numpy way would just look like this:
        # _dim = inv(_dm)
        _tmp = _dm[0,0] * _dm[1,1] - _dm[0,1] * _dm[1,0]
        _det = 1 / _tmp
        _dim[0,0] = _det * _dm[1,1]
        _dim[0,1] = -1 * _det * _dm[0,1]
        _dim[1,0] = -1 * _det * _dm[1,0]
        _dim[1,1] = _det * _dm[0,0]

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

        _pm = np.zeros((2,2),dtype=complex)
        ci = 0+1j
        a = -1 * ci * phil
        b = ci * phil

        _pm[0,0] = np.exp(a)
        _pm[1,1] = np.exp(b)

        return _pm


