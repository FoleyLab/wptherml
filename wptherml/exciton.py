import numpy as np
from .spectrum_driver import SpectrumDriver


class ExcitonDriver(SpectrumDriver):
    """A class for computing the dynamics and spectra of a system modelled by the Frenkel Exciton Hamiltonian

    Attributes
    ----------
    radius : float
        the radius of the sphere


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
        
        # Probably want to allow the user to specify an initial state!
        # but right now just have the initial state with exciton localized on site 1
        self.c_vector[0,0] = 1 + 0j
        #self.density_matrix = np.dot(self.c_vector, np.conj(self.c_vector.T))
        self.density_matrix = np.dot(self.c_vector, np.conj(self.c_vector.T))

    def parse_input(self, args):
        if "exciton_energy" in args:
            self.exciton_energy = args["exciton_energy"]
        else:
            self.exciton_energy = 0.5
        if "number_of_monomers" in args:
            self.number_of_monomers = args["number_of_monomers"]
        else:
            self.number_of_monomers = 2
        if "displacement_between_monomers" in args:
            self.displacement_between_monomers = args["displacement_between_monomers"]
        else:
            self.displacement_between_monomers = np.array([1, 0, 0])

        if "transition_dipole_moment" in args:
            self.transition_dipole_moment = args["transition_dipole_moment"]
        else:
            self.transition_dipole_moment = np.array([0, 0, 1])
        if "refractive_index" in args:
            self.refractive_index = args["refractive_index"]
        else:
            self.refractive_index = 1
        
        if "vert_displacement_between_monomers" in args: 
            self.vert_displacement_between_monomers = args["vert_displacement_between_monomers"]
        else: 
            self.vert_displacement_between_monomers = np.array([0, 19.633983, 0])
        
        if "diag_displacement_between_monomers" in args:
            self.diag_displacement_between_monomers = args["diag_displacement_between_monomers"]
        else:
            self.diag_displacement_between_monomers = np.array([-17.7348345, 19.633983, 0])
        


        self.coords = np.zeros((3, self.number_of_monomers))

        for i in range(self.number_of_monomers):
            self.coords[:, i] = self.displacement_between_monomers * i
        
        self.wvlngth_variable = np.arange(0, 400.001, 0.01)

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
        _r_vec = self.coords[:, m] - self.coords[:, n]

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
    
    def _compute_2D_dd_coupling(self, r_vector):
        """ Function that computes the dipole dipole coupling contribution of the total energy of a system based upon mu_d,
            the transition dipole moment of the donor, mu_a, the transition dipole moment of the acceptor, r_vector, the 
            distance separating the donor and acceptor, and the refractive index, a paremeter that describes the effect of 
            the system on light.
    
        Arguments
        ---------
        mu_d : numpy array of floats
         the transition dipole mooment of the donor
        mu_a : numpy array of floats
            the transition dipole moment of the acceptor
        r : numpy array of floats
            the distance separating the donor and acceptor
        n : float
            refractive index of the medium
        """

        r_scalar = np.sqrt(np.dot(r_vector, r_vector))
    
        return (1 / (self.refractive_index ** 2 * r_scalar ** 3)) * (np.dot(self.transition_dipole_moment, self.transition_dipole_moment) - 3 * (np.dot(self.transition_dipole_moment, r_vector) * np.dot(r_vector, self.transition_dipole_moment)) / r_scalar ** 2)

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

        # nested loop to build Hamiltonian
        for _n in range(_N):
            for _m in range(_N):
                # <== call _compute_H0_element and store value -> H0
                H0 = self._compute_H0_element(_n, _m)  # <== Note self. notation
                # <== call _compute_dipole_dipole_coupling and store value -> V
                V = self._compute_dipole_dipole_coupling(
                    _n, _m
                )  # <== Note self. notation
                # <== assign H0 + V to appropriate element of self.exciton_hamiltonian
                self.exciton_hamiltonian[_n, _m] = (
                    H0 + V
                )  # <= Note we will store the elements in hamiltonian attribute
    
    def _find_indices(self, mon_int):
        """Helper function to designate indices of a a matrix element designated by an integer
    
        Arguments
        ---------
        mons : int
            Integer value that creates n x n array that represents film dimentions
        mon_int : int
            Integer value thatg will find a target integer within the matrix which corresponds to a set of indices
        """
        film_matrix = np.zeros((self.number_of_monomers, self.number_of_monomers))
        rows = self.number_of_monomers
        cols = self.number_of_monomers
        mon_range = range(self.number_of_monomers ** 2)
        for r in range(rows):
            for c in range(cols):
                mon_idx = r * cols + c
                film_matrix[r][c] = mon_range[mon_idx]
    
        indices = np.column_stack(np.where(film_matrix == mon_int))
        return indices

    def build_2D_hamiltonian(self):
        """ Function that builds the Hamailtonian which models the time evolution of an excitonic system based upon the
        field free energy of the system and the dipole dipole coupling of the sysetem
         
        Attribute
        ---------
        exciton_hamiltonian : number_of_monomers x number_of_monomers numpy array of floats
            the exciton Hamiltonian, initialized by init and to-be-filled with appropriate values
            by this function
        
        """
        _N = self.number_of_monomers
        self.exciton_hamiltonian_2D = np.zeros((_N ** 2, _N ** 2))
        self.dd_p1 = self._compute_2D_dd_coupling(self.vert_displacement_between_monomers) * (9.8 / 8.8)
        self.dd_n1 = self._compute_2D_dd_coupling(self.diag_displacement_between_monomers) * (9.8 / 8.8)
        for _n in range(_N ** 2):
            for _m in range(_N ** 2):
                H0 = self._compute_H0_element(_n, _m)
                if np.all(self._find_indices(_n) == self._find_indices(_m) + np.array([-1, -1])): V = self.dd_n1
                elif np.all(self._find_indices(_n) == self._find_indices(_m) + np.array([1, 1])): V = self.dd_n1
                elif np.all(self._find_indices(_n) == self._find_indices(_m) + np.array([-1, 0])): V = self.dd_p1
                elif np.all(self._find_indices(_n) == self._find_indices(_m) + np.array([1, 0])): V = self.dd_p1
                else: V = 0
                self.exciton_hamiltonian_2D[_n, _m] = (
                    H0 + V
                )
        return self.exciton_hamiltonian_2D

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
        _dx = self.coords[0, 1] - self.coords[0, 0]

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
        _x_max = self.coords[0, _N_max] + 3 * _fwhm
        # create the x-grid from 0 to _x_max
        _len = 500
        self.x = np.linspace(-_dx, _x_max, _len)

        self.phi = np.zeros((_len, self.number_of_monomers))

        for n in range(self.number_of_monomers):
            _x_n = self.coords[0, n]
            self.phi[:, n] = _a * np.exp(-((self.x - _x_n) ** 2) / (2 * _c**2))

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

    def _2D_rk_exciton(self, dt):
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
        k_1 = -ci * np.dot(self.exciton_hamiltonian_2D, self.c_vector)
        k_2 = -ci * np.dot(self.exciton_hamiltonian_2D, (self.c_vector + k_1 * dt / 2))
        k_3 = -ci * np.dot(self.exciton_hamiltonian_2D, (self.c_vector + k_2 * dt / 2))
        k_4 = -ci * np.dot(self.exciton_hamiltonian_2D, (self.c_vector + k_3 * dt))
        self.c_vector = self.c_vector + (1 / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4) * dt

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
        

        return 1 ** 2 / ((self.wvlngth_variable - lambda_0) ** 2 + 1 ** 2)

    def compute_spectrum(self, wavelengths):
        """Method that will return an array of values corresponding to a plotable spectrum

        Arguments
        ---------
        wavelengths: numpy array of floats 

        """
        result = np.zeros_like(self.wvlngth_variable)
        for x0 in zip(wavelengths):
            result += self.lorentzian(x0)
        
        return result