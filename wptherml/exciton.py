import numpy as np
from matplotlib import pyplot as plt
from .spectrum_driver import SpectrumDriver


class ExcitonDriver(SpectrumDriver):
    """A class for computing the dynamics and spectra of a system modelled by the Frenkel Exciton Hamiltonian

    Attributes
    ----------
    radius : float
        the radius of the sphere
    
    c_vector : numpy array of flaots
        vector coefficient associated with the weight of a state in an expansion of the wavefunction

    density_matrix : numpy array of floats
        Adjoint of a c_vector multiplied by the original c_vector

    exciton_energy : float
        energy of the monomer exciton in atomic units

        aggregate_shape : tuple
            The number of monomers along each coordinate (Nx, Ny, Nz)

    displacement_vector : numpy array of flaots
        displacement between each monomer in cartesian coordinates

    transition_dipole_moment : numpy array of floats
        dipole associated with the transition from the ground state to the excited state in atomic units

    refractice_index : float
        refractive index of the monomer film


    

    Returns
    -------
    None

    Examples
    --------
    >>> fill_in_with_actual_example!
    """

    def __init__(self, args):
        self.parse_input(args)
        # print("Exciton Energy is  ", self.exciton_energy)
        # allocate the exciton Hamiltonian

        self.exciton_hamiltonian = np.zeros(
            (self.number_of_monomers, self.number_of_monomers)
        )
        self.c_vector = np.zeros(
            (self.number_of_monomers, 1),  dtype=complex
        )  # <== wavefunction coefficient vector
        
        # define cartesian coordinates of each monomer
        self._compute_cartesian_coordinates()

        # build exciton Hamiltonian
        self.build_exciton_hamiltonian()

        # Probably want to allow the user to specify an initial state!
        # but right now just have the initial state with exciton localized on site 1
        self.c_vector[0,0] = 1 + 0j
        self.density_matrix = np.dot(self.c_vector, np.conj(self.c_vector.T))

    def parse_input(self, args):
        if "exciton_energy" in args:
            self.exciton_energy = args["exciton_energy"]
        else:
            self.exciton_energy = 0.5

        if "aggregate_shape" in args:
            self.aggregate_shape = args["aggregate_shape"]

        # default shape is (2, 2, 1) -> 4 monomers total, 2 monomers displaced along x-axis and 2 along y-axis, all in plane (2D)
        else:
            self.aggregate_shape = (2, 2, 1)
        
        # user might accidentally make one element of the tuple zero
        if self.aggregate_shape[0] == 0 or self.aggregate_shape[1] == 0 or self.aggregate_shape[2] == 0:
            print(" Invalid shape!  All directions must have at least 1 layer!")
            print(" Smallest possible shape (1,1,1) is a monomer - please check your input and try again")
            

        self.number_of_monomers = self.aggregate_shape[0] * self.aggregate_shape[1] * self.aggregate_shape[2]
        # allow the user to specify the displacements along each axis (x, y, z) as a vector
        if "displacement_vector" in args:
            self.displacement_vector = args["displacement_vector"]

        # default -> displacement along x = 35.47 a.u, displacement along y = 19.63 a.u., displacement along z = 8.47
        else:
            self.displacement_vector = [35.47, 19.63, 8.47]

        if "transition_dipole_moment" in args:
            self.transition_dipole_moment = args["transition_dipole_moment"]
        else:
            self.transition_dipole_moment = np.array([0, 0, 1])

        if "refractive_index" in args:
            self.refractive_index = args["refractive_index"]
        else:
            self.refractive_index = 1

        self.wvlngth_variable = np.arange(0, 650.001, 0.01)
        self.number_of_monomers = self.aggregate_shape[0] * self.aggregate_shape[1] * self.aggregate_shape[2]
    def _compute_cartesian_coordinates(self):
        """ Method to assign cartesian coordinates to the monomers
            
        Arguments
        ---------
        None

        Attributes
        ----------
        aggregate_shape : tuple
            The number of monomers along each coordinate (Nx, Ny, Nz)

        displacement_vector : numpy array of floats
            The displacement along each coordinate [Dx, Dy, Dz]

        number_of_monomers : int
            The total number of monomers

        coords : number_of_monomers x 3 numpy array of floats
            X, Y, Z coordinates for each monomer        
        """

        # get number of monomers displaced along x, y, and z directions
        _Nx = self.aggregate_shape[0]
        _Ny = self.aggregate_shape[1]
        _Nz = self.aggregate_shape[2]
        
        # initialize coords
        self.coords = np.zeros((self.number_of_monomers, 3))

        _Nc = 0
        for i in range(_Nx):
            for j in range(_Ny):
                for k in range(_Nz):
                    self.coords[_Nc, 0] = i * self.displacement_vector[0]
                    self.coords[_Nc, 1] = j * self.displacement_vector[1]
                    self.coords[_Nc, 2] = k * self.displacement_vector[2]
                    _Nc += 1
        return self.coords
    

    def _compute_H0_element(self, n, m):
        """Method to compute the matrix elements of H0

        Arguments
        ---------
        n : int
            the index of site n 
        m : int
            the index of site m 

        Returns
        -------
        H_nm : float
            The matrix elements corresponding to the interactions between sites n and m
        """
     
        H_nm = self.exciton_energy * (n == m)
        return H_nm

    def _compute_dipole_dipole_coupling(self, n, m):
        """Method to compute the dipole-dipole potential between excitons located on site n and site m

        Arguments
        ---------
        n : int
            the index of site n 
        m : int
            the index of site m 

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

        # calculate separation vector between site m and site n
        self.coord_array = self._compute_cartesian_coordinates()
        _r_vec = self.coord_array[n,:] - self.coord_array[m,:]

        # self.transition_dipole_moment is the transition dipole moment!
        if n != m:
            V_nm = (
                1 / (self.refractive_index**2 * np.sqrt(np.dot(_r_vec, _r_vec)) ** 3)
            ) * (
                np.dot(self.transition_dipole_moment, self.transition_dipole_moment)
                - 3
                * (
                    (
                        np.dot(self.transition_dipole_moment, _r_vec)
                        * np.dot(_r_vec, self.transition_dipole_moment)
                    )
                    / (np.sqrt(np.dot(_r_vec, _r_vec)) ** 2)
                )
            )
        else:
            V_nm = 0

        return V_nm
    

    def build_exciton_hamiltonian(self):
        """Method to build the Frenkel Exciton Hamiltonian

        Attribute
        ---------
        exciton_hamiltonian : number_of_monomers x number_of_monomers numpy array of floats
            the exciton Hamiltonian, initialized by init and to-be-filled with appropriate values
            by this function

        Notes
        -----

        """
        _N = self.number_of_monomers  # <== _N is just easier to type!
        self.exciton_hamiltonian = np.zeros((self.number_of_monomers, self.number_of_monomers))
        # nested loop to build Hamiltonian
        for _n in range(_N):
            for _m in range(_N):
                # <== call _compute_H0_element and store value -> H0
                H0 = self._compute_H0_element(_n, _m)  # <== Note self. notation
                # <== call _compute_dipole_dipole_coupling and store value -> V
                V = self._compute_dipole_dipole_coupling(
                    _n, _m
                )   # <== Note self. notation
                # <== assign H0 + V to appropriate element of self.exciton_hamiltonian
                self.exciton_hamiltonian[_n, _m] = (
                    H0 + V 
                )  # <= Note we will store the elements in hamiltonian attribute
        return self.exciton_hamiltonian
        

    """def build_2D_hamiltonian(self):
        Function that builds the Hamailtonian which models the time evolution of an excitonic system based upon the
        field free energy of the system and the dipole dipole coupling of the sysetem
         
        Attribute
        ---------
        exciton_hamiltonian : number_of_monomers x number_of_monomers numpy array of floats
            the exciton Hamiltonian, initialized by init and to-be-filled with appropriate values
            by this function
        
        
        _N = self.number_of_monomers
        exciton_hamiltonian_2D = np.zeros((_N, _N))
        for _n in range(_N):
            for _m in range(_N):
                H0 = self._compute_H0_element(_n, _m)
                V = self._compute_dipole_dipole_coupling(_n, _m)
                exciton_hamiltonian_2D[_n, _m] = (
                    H0 + V
                )
        return exciton_hamiltonian_2D
"""
    def compute_exciton_wavefunction_site_basis(self):
        """
        Will compute the single-exciton wavefunctions (approximated as Gaussians) for each site.

        ***Note:***  This is probably not the best model... better model might be a Slater function, but
        we can fix that later. Also right now the width is somewhat arbitrary.  In reality, the exciton
        has a Bohr radius that will determine the width.  We will update this later as well!

        Arguments
        ---------
        n : int
            the site index offset from python index by +1

        Attributes
        ----------
        x : 1 x _len numpy array of floats
            the spatial grid that we will evaluate the exciton wavefunction on

        x_min : float
            the minimum x-value on the grid x

        x_max : float
            the maximum x-value on the grid x

        phi : _len x number_of_monomer numpy array of floats
            the single-exciton wavefunctions for each site

        Note: self.phi[:,0] -> exciton wavefunction for site n = 1
              self.phi[:,1] -> exciton wavefunction for site n = 2
              etc.

        """
        # get distance between sites
        _dx = self.coords[1, 0] - self.coords[0, 0]

        # full-width at half-max based on distance between sites
        _fwhm = _dx / 2

        # width parameter of the Gaussian
        _c = _fwhm / 2.35482

        # normalization of Gaussian
        _a = 1 / (_c * np.sqrt(2 * np.pi))

        # create grid of x values spanning all sites in the system
        # get largest site index
        _N_max = self.number_of_monomers - 1
        # get x-value associated with largest site index
        _x_max = self.coords[_N_max, 0] + 3 * _fwhm
        # create the x-grid from 0 to _x_max
        _len = 500
        self.x = np.linspace(-_dx, _x_max, _len)

        self.phi = np.zeros((_len, self.number_of_monomers))

        for n in range(self.number_of_monomers):
            _x_n = self.coords[n, 0]
            self.phi[:, n] = np.exp(-((self.x - _x_n) ** 2) / (2 * _c**2))

        self.x_max = _x_max
        self.x_min = -_dx

    def _rk_exciton(self, dt):
        """Function that will take c(t0) and H and return c(t0 + dt)

        Arguments
        ---------
        dt : float
            the increment in time in atomic units

        Attributes
        ----------
        exciton_hamiltonian : NxN numpy array of floats
            the Hamiltonian matrix that drives the dynamics

        c_vector : 1xN numpy array of complex floats
            the current wavefunction vector that will be updated

        """
        ci = 0 + 1j
        k_1 = -ci * np.dot(self.exciton_hamiltonian, self.c_vector)
        k_2 = -ci * np.dot(self.exciton_hamiltonian, (self.c_vector + k_1 * dt / 2))
        k_3 = -ci * np.dot(self.exciton_hamiltonian, (self.c_vector + k_2 * dt / 2))
        k_4 = -ci * np.dot(self.exciton_hamiltonian, (self.c_vector + k_3 * dt))
        self.c_vector = self.c_vector + (1 / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4) * dt
        return self.c_vector
    def _rk_exciton_density_matrix(self, dt):
        """Function that will take D and H and return D(t0 + dt)

        Arguments
        ---------
        dt : float
            the increment in time in atomic units

        Attributes
        ----------
        exciton_hamiltonian : NxN numpy array of floats
            the Hamiltonian matrix that drives the dynamics

        density_matrix : NxN numpy array of complex floats
            the current density matrix that will be updated

        """
        ci = 0 + 1j
        # going to make some temporary arrays for the partial updates of d
        # to make this more readable
        # also note I am using np.copy() here rather than setting _H = self.exciton_hamiltonian
        # see here for why: https://stackoverflow.com/questions/27538174/why-do-i-need-np-array-or-np-copy
        _H = np.copy(self.exciton_hamiltonian)
        _d0 = np.copy(self.density_matrix)

        k_1 = -ci * (np.dot(_H, _d0) - np.dot(_d0, _H))
        _d1 = _d0 + k_1 * dt / 2

        k_2 = -ci * (np.dot(_H, _d1) - np.dot(_d1, _H))
        _d2 = _d0 + k_2 * dt / 2

        k_3 = -ci * (np.dot(_H, _d2) - np.dot(_d2, _H))
        _d3 = _d0 + k_3 * dt

        k_4 = -ci * (np.dot(_H, _d3) - np.dot(_d3, _H))

        # final update - using np.copy() again
        self.density_matrix = _d0 + 1 / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4) * dt
        
        return self.density_matrix

    def msd_psi(self, dt, N_time):
        """Method that will take dt and a number of time steps and return the mean squared displacement

        Arguments
        ---------
        dt : float
            the increment in time in atomic units
    
        N_time : int
            the number of time steps
    
         """
        prod1 = 0
        prod2 = 0
        msd_matrix = np.zeros((1, N_time))
        for i in range(self.number_of_monomers):
            for j in range(self.number_of_monomers):
                x_int = np.conj(self.c_vector[i] * self.phi[:,i]) * self.x * self.c_vector[j] * self.phi[:,j]
                x_o = np.trapz(x_int, self.x)
                for k in range(N_time):
                    self.c_temp = self._rk_exciton(dt)
                    prod1 += np.conj(self.c_temp[i] * self.phi[:,i]) * self.x ** 2 * self.c_temp[j] * self.phi[:,j]
                    prod2 += np.conj(self.c_temp[i] * self.phi[:,i]) * self.x * self.c_temp[j] * self.phi[:,j]
                    msd_matrix[0,k] = np.trapz(prod1, self.x) - 2 * x_o * np.trapz(prod2, self.x) + x_o ** 2
    
        return msd_matrix[0,:]
    
    def msd_density_matrix(self, dt, N_time):
        """Method that will take dt and a number of time steps and return the mean squared displacement

        Arguments
        ---------
        dt : float
            the increment in time in atomic units
    
        N_time : int
            the number of time steps
    
         """
        x_matrix = np.zeros((self.number_of_monomers, self.number_of_monomers))
        msd_matrix = np.zeros(N_time)

        for n in range(self.number_of_monomers):
            for m in range(self.number_of_monomers):
                x_nm = self.phi[:,m] * self.x * self.phi[:,n]
                x_matrix[n,m] = np.trapz(x_nm)
        
        x_o = np.trace(x_matrix * self.density_matrix)

        for i in range(self.number_of_monomers):
            for j in range(self.number_of_monomers):
                for k in range(N_time):
                    density_matrix = self._rk_exciton_density_matrix(dt)
                    prod1 = np.trace(density_matrix * (x_matrix ** 2))
                    prod2 = np.trace(density_matrix * x_matrix)
                    msd_matrix[k] = prod1 - 2 * x_o * prod2 * x_o ** 2
        
        return msd_matrix


    def lorentzian(self, lambda_0):
        """Helper method that will take an x array and return a lorentzian function corresponding to that x array

        Arguments
        ---------
        lambda_0 : np array of floats, 
         """
        

        return 5 ** 2 / ((self.wvlngth_variable - lambda_0) ** 2 + 5 ** 2)

    def spectrum_array(self):
        """Method that will return an array of values corresponding to a plotable spectrum

        """

        test_eigenvalues = np.linalg.eigh(self.build_exciton_hamiltonian())

        Hartree_to_J = 4.35974 * 10 ** (-18)
        h = 6.626 * 10 ** (-34)
        lightspeed = 2.998 * 10 ** 8
        m_to_nm = 10 ** 9

        eigh_J = test_eigenvalues.eigenvalues * Hartree_to_J
        eigh_wvl = m_to_nm * h * lightspeed / eigh_J

        abs_spec = np.zeros_like(self.wvlngth_variable)
        for x0 in zip(eigh_wvl):
            abs_spec += self.lorentzian(x0)
        
        return abs_spec
    
    def compute_spectrum(self):
        """method that will take values computed from spectrum_array and plot them vs wavelength
    
        """
        test_spec = self.spectrum_array()
        spectrum_plot = plt.plot(self.wvlngth_variable, test_spec, 'b-')

        return spectrum_plot 
    
