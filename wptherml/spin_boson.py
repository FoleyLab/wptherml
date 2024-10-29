import numpy as np
from numpy import linalg as la 
from matplotlib import pyplot as plt
from .spectrum_driver import SpectrumDriver


class SpinBosonDriver(SpectrumDriver):
    """A class for computing the dynamics and spectra of coupled exciton-boson (e.g. QD - plasmon, exciton-polariton, etc) systems using
       the spin boson for N 2-level systems coupled to an N'-level Harmonic oscillator.  For conventions of exciton states and their
       ladder operators, see here: https://www.phys.hawaii.edu/~yepez/Spring2013/lectures/Lecture2_Quantum_Gates_Notes.pdf

       Important: We order the basis as |s> \otimes |q1> \otimes |q2> ... \otimes |qn>

    Attributes
    ----------
    number_of_excitons : int
        number of excitonic subsystems

    number_of_boson_levels : int
        number of boson levels

    exciton_energy_ev : float
        energy of each exciton subsystem in eV

    boson_energy_ev : float
        fundamental energy of the boson subsystem in eV

    exciton_energy_au : float
        energy of each exciton subsystem in atomic units

    boson_energy_au : float
        fundamental energy of the boson subsystem in atomic units

    exciton_boson_coupling_ev : float
        coupling between each exciton subsystem and the boson subsystem in eV

    exciton_boson_coupling_au : float
        coupling between each exciton subsystem and the boson subsystem in atomic units

    boson_dipole_magnitude_au : float
        transition dipole moment of the bosonic subsystem

    exciton_transition_dipole_magnitude_au : float
        transition dipole moment of the excitonic subsystems

    exciton_ground_state_dipole_magnitude_au : float
        permanent dipole moment of the excitonic ground state

    exciton_excited_state_dipole_magnitude_au : float
        permanent dipole moment of the excitonic excited state

    single_exciton_basis : numpy matrix
        basis states for a single excition

    n_exciton_basis : numpy matrix
        basis states for the collection of N excitons

    boson_basis : numpy matrix
        basis states for the N-level Harmonic oscillator

    exciton_boson_basis : numpy matrix
        basis states for the collection of N excitons and the N'-level Harmonic oscillator


    Returns
    -------
    None

    Examples
    --------
    >>> fill_in_with_actual_example!
    """

    def __init__(self, args):
        """constructor for the SpinBosonDriver class"""
        # make sure all keys are lowercase only
        args = {k.lower(): v for k, v in args.items()}

        # conversion from eV to atomic units
        self.ev_to_au = 3.6749322175665e-2

        # parse user inputs
        self.parse_input(args)

        # compute spectrum
        self.compute_spectrum()

    def parse_input(self, args):
        if "number_of_excitons" in args:
            self.number_of_excitons = args["number_of_excitons"]
        else:
            self.number_of_excitons = 1

        if "number_of_boson_levels" in args:
            self.number_of_boson_levels = args["number_of_boson_levels"]
        else:
            self.number_of_boson_levels = 2  # includes |0> and |1>

        if "exciton_energy_ev" in args:
            self.exciton_energy_ev = args["exciton_energy_ev"]
        else:
            self.exciton_energy_ev = 1.0

        if "boson_energy_ev" in args:
            self.boson_energy_ev = args["boson_energy_ev"]
        else:
            self.boson_energy_ev = 1.0

        if "exciton_boson_coupling_ev" in args:
            self.exciton_boson_coupling_ev = args["exciton_boson_coupling_ev"]
        else:
            self.exciton_boson_coupling_ev = 0.01

        if "boson_dipole_magnitude_au" in args:
            self.boson_dipole_magnitude_au = args["boson_dipole_magnitude_au"]
        else:
            self.boson_dipole_magnitude_au = 1000.

        if "exciton_transition_dipole_magnitude_au" in args:
            self.exciton_transition_dipole_magnitude_au = args["exciton_transition_dipole_magnitude_au"]
        else:
            self.exciton_transition_dipole_magnitude_au = 10.

        if "exciton_ground_state_dipole_magnitude_au" in args:
            self.exciton_ground_state_dipole_magnitude_au = args["exciton_ground_state_dipole_magnitude_au"]
        else:
            self.exciton_ground_state_dipole_magnitude_au = 10.

        if "exciton_excited_state_dipole_magnitude_au" in args:
            self.exciton_excited_state_dipole_magnitude_au = args["exciton_excited_state_dipole_magnitude_au"]
        else:
            self.exciton_excited_state_dipole_magnitude_au = 10.
        

        # convert energies from eV to au
        self.exciton_energy_au = self.exciton_energy_ev * self.ev_to_au
        self.boson_energy_au = self.boson_energy_ev * self.ev_to_au
        self.exciton_boson_coupling_au = self.exciton_boson_coupling_ev * self.ev_to_au

    def build_boson_basis(self):
        """build the basis for the N-level Harmonic oscillator

        Args
        ------
        None

        Attributes
        ----------
        number_of_boson_levels : int
            number of boson levels

        boson_basis : numpy matrix
            basis states for the N-level Harmonic oscillator

        Returns
        -------
        None

        """
        self.boson_basis = np.eye(self.number_of_boson_levels)

    def build_exciton_basis(self):
        """build the basis for the N excitonic subsystems

        Args
        ------
        None

        Attributes
        ----------
        number_of_excitons : int
            number of excitonic subsystems

        single_exciton_basis : numpy matrix
            basis states for a single excition

        exciton_basis_dimension : int
            dimension of the N-exciton hilbert space

        n_exciton_basis : numpy matrix
            basis states for the collection of N excitons

        Returns
        -------
        None
        """
        self.single_exciton_basis = np.array([[1, 0], [0, 1]])  #np.matrix("1 0 ; 0 1")
        self.exciton_basis_dimension = 2**self.number_of_excitons
        self.n_exciton_basis = np.eye(self.exciton_basis_dimension)

    def build_exciton_boson_basis(self):
        """build the basis for the N excitonic subsystems and the N'-level Harmonic oscillator in order
            |s> \otimes |q_1> \otimes |q_2> \otimes ... \otimes |q_N>

        Arguments
        ----------
        None

        Attributes
        ----------
        n_exciton_basis : numpy matrix
            basis states for the collection of N excitons

        boson_basis : numpy matrix
            basis states for the N'-level Harmonic oscillator

        exciton_boson_basis : numpy matrix
            basis states for the collection of N excitons and the N'-level Harmonic oscillator in order
            |s> \otimes |q_1> \otimes |q_2> \otimes ... \otimes |q_N>

        Returns
        -------
        None

        """
        self.exciton_boson_basis = np.kron(self.boson_basis, self.n_exciton_basis)

    def build_bosonic_ladder_operators(self):
        """build the bosonic raising and lowering operators

        Arguments
        ----------
        None

        Attributes
        ----------
        number_of_boson_levels : int
            number of boson levels

        b_matrix : numpy matrix
            matrix representation of the lowering operator

        b_dagger_matrix : numpy matrix
            matrix representation of the raising operator


        Returns
        -------
        None

        """
        self.b_matrix = np.zeros(
            (self.number_of_boson_levels, self.number_of_boson_levels)
        )
        for i in range(1, self.number_of_boson_levels):
            self.b_matrix[i - 1, i] = np.sqrt(i)

        self.b_dagger_matrix = self.b_matrix.transpose().conjugate()

    def build_boson_energy_operator(self):
        """build the boson energy operator in the N'-level bosonic - N-qd coupled Hilbert space

        Arguments
        ----------
        None

        Attributes
        ----------

        boson_energy_au : float
            fundamental energy of the boson subsystem in atomic units

        b_matrix : numpy matrix
            matrix representation of the lowering operator

        b_dagger_matrix : numpy matrix
            matrix representation of the raising operator

        boson_number_operator : numpy matrix
            matrix representation of the bosonic number operator in the N'-level bosonic Hilbert space

        boson_energy_operator : numpy matrix
            matrix representation of the bosonic energy operator in the N-excitonic N'-level bosonic Hilbert space

        Returns
        -------
        None
        """
        # build number operator in the N'-level bosonic Hilbert space
        self.boson_number_operator = np.dot(self.b_dagger_matrix, self.b_matrix)

        # create the energy operator on the boson Hilbert space: \hbar \omega (\hat{N} + 1/2 I_S)
        _energy_operator_on_boson_space = (
            self.boson_energy_au * self.boson_number_operator
            + 0.5 * self.boson_energy_au * np.eye(self.number_of_boson_levels)
        )
        #print("Printing energy operator on boson space")
        #print(_energy_operator_on_boson_space)

        # build the boson energy operator in the coupled Hilbert space
        self.boson_energy_operator = np.kron(
            _energy_operator_on_boson_space, self.n_exciton_basis
        )

    def build_boson_dipole_operator(self):
        """build the boson energy operator in the N-qd N'-level coupled Hilbert space

        Arguments
        ----------
        None

        Attributes
        ----------

        boson_dipole_magnitude_au : float
            fundamental energy of the boson subsystem in atomic units

        b_matrix : numpy matrix
            matrix representation of the lowering operator

        b_dagger_matrix : numpy matrix
            matrix representation of the raising operator

        boson_dipole_operator : numpy matrix
            matrix representation of the bosonic energy operator in the N-excitonic N'-level bosonic Hilbert space

        Returns
        -------
        None
        """
        # build number operator in the N'-level bosonic Hilbert space
        _boson_dipole_operator_on_boson_space = self.boson_dipole_magnitude_au * (self.b_dagger_matrix + self.b_matrix)


        # build the boson energy operator in the coupled Hilbert space
        self.boson_dipole_operator = np.kron(
            _boson_dipole_operator_on_boson_space, self.n_exciton_basis
        )


    def build_operator_for_exciton_j(self, j, operator="sigma_z", factor = 1):
        """build operator for the j-th exciton in the coupled N-exciton hilbert space

        Arguments
        ----------
        operator : string
            operator to build

        j : int
            index of the exciton; start couunt from 0

        factor : float
            scaling factor for the operator, defaults to 1

        Attributes
        ----------
        exciton_operator_j

        Returns
        -------
        None

        """
        # define the operator on the single exciton hilbert space that we
        # wish to use in the composite hilbert space
        if operator == "sigma_z":
            self.single_exciton_operator = np.array([[1, 0], [0, -1]]) * factor  #np.matrix("1 0 ; 0 -1")

        elif operator == "sigma_x":
            self.single_exciton_operator = np.array([[0, 1], [1, 0]]) * factor #np.matrix("0 1 ; 1 0")

        elif operator == "sigma_y":
            self.single_exciton_operator = np.array([[0, -1j], [1j, 0]]) * factor #np.matrix("0 -1j ; 1j 0")

        elif operator == "sigma_p":  # sigma_p |0> == sigma_p [1 0].T = |1> == [0 1].T
            self.single_exciton_operator = np.array([[0, 0], [1, 0]]) * factor #np.matrix("0 0 ; 1 0")

        elif operator == "sigma_m":  # sigma_m |1> == sigma_m [0 1].T = |0> == [1 0].T
            self.single_exciton_operator = np.array([[0, 1], [0, 0]]) * factor #np.matrix("0 1 ; 0 0")

        elif operator == "transition_dipole_operator":
            self.single_exciton_operator = self.exciton_transition_dipole_magnitude_au * np.array([[0, 1], [1, 0]]) * factor #np.matrix("0 1 ; 1 0")

        elif operator == "total_dipole_operator":
            self.single_exciton_operator = np.array([[self.exciton_ground_state_dipole_magnitude_au, self.exciton_transition_dipole_magnitude_au],
                                                     [self.exciton_transition_dipole_magnitude_au, self.exciton_excited_state_dipole_magnitude_au]]) * factor
        elif operator == "sigma_pm":
            self.single_exciton_operator = np.array([[0, 0], [0, 1]]) * factor #np.matrix("0 0 ; 0 1")

        else:
            # if no valid option given, use an identity
            self.single_exciton_operator = np.array([[1, 0], [0, 1]]) * factor #np.matrix("1 0 ; 0 1")

        if self.number_of_excitons == 1:
            # exciton_operator_j is just a single exciton operator
            self.exciton_operator_j = self.single_exciton_operator

        elif self.number_of_excitons == 2:
            # exciton_operator_j is either op(1) x Iq(2) or Iq(1) x op(2)
            _ID_q = np.eye(2)
            if j == 0:
                self.exciton_operator_j = np.kron(self.single_exciton_operator, _ID_q)
            elif j == 1:
                self.exciton_operator_j = np.kron(_ID_q, self.single_exciton_operator)

        elif self.number_of_excitons > 2:
            # need to first see what j is to determine structure of exciton_operator_j
            if j == 0:
                # exciton_operator_j is op(1) x Iq(2,..,N)
                dim = 2 ** (self.number_of_excitons - 1)
                _ID_q = np.eye(dim)
                self.exciton_operator_j = np.kron(self.single_exciton_operator, _ID_q)

            elif j == self.number_of_excitons - 1:
                # exciton_operator_j is Iq(1,...,N-1) x op(N)
                dim = 2 ** (self.number_of_excitons - 1)
                _ID_q = np.eye(dim)
                self.exciton_operator_j = np.kron(_ID_q, self.single_exciton_operator)

            else:  # exciton_operator_j is Iq(1,...,j-1) x op(j) x Iq(j+1,...,N)
                dim_L = 2**j
                dim_R = 2 ** (self.number_of_excitons - j - 1)
                _ID_q_L = np.eye(dim_L)
                _ID_q_R = np.eye(dim_R)
                self.exciton_operator_j = np.kron(
                    _ID_q_L, np.kron(self.single_exciton_operator, _ID_q_R)
                )

    def build_exciton_energy_operator(self):
        """compute the exciton energy operator in the N-exciton N'-level bosonic Hilbert space

        Arguments
        ----------
        None

        Attributes
        ----------
        exciton_energy_au : float
            energy of each exciton subsystem in atomic units

        exciton_energy_operator : numpy matrix
            dim x dim x N tensor representation of the exciton energy operator in the N-exciton hilbert space
            where dim is the size of the N-exciton N'-level bosonic hilbert space


        Returns
        -------
        None

        """
        # dimension of the coupled Hilbert space
        _dim = self.exciton_boson_basis.shape[0]

        # identity on the boson Hilbert space
        _Is = np.eye(self.number_of_boson_levels)

        # create tensor for exciton operators
        self.exciton_energy_operator = np.zeros((_dim, _dim))

        for i in range(self.number_of_excitons):
            # get the sigma_+ \sigma_- operator for the ith exciton in the N-exciton Hilbert space
            self.build_operator_for_exciton_j(i, "sigma_pm", factor = self.exciton_energy_au)

            # take tensor product of the identity on the boson Hilbert space with this exciton operator
            _Op = np.kron(_Is, self.exciton_operator_j)

            # add this operator to exciton_energy_operator
            self.exciton_energy_operator  += _Op

     

    def build_exciton_dipole_operator(self):
        """compute the exciton dipole operator in the N-exciton N'-level bosonic Hilbert space

        Arguments
        ----------
        None

        Attributes
        ----------

        exciton_dipole_operator : numpy matrix
            dim x dim x N tensor representation of the exciton energy operator in the N-exciton hilbert space
            where dim is the size of the N-exciton N'-level bosonic hilbert space


        Returns
        -------
        None

        """
        # dimension of the coupled Hilbert space
        _dim = self.exciton_boson_basis.shape[0]
        # identity on the boson Hilbert space
        _Is = np.eye(self.number_of_boson_levels)

        # create tensor for exciton operators
        self.exciton_dipole_operator = np.zeros((_dim, _dim))

        for i in range(self.number_of_excitons):
            # get the sigma_+ \sigma_- operator for the ith exciton in the N-exciton Hilbert space
            self.build_operator_for_exciton_j(i, "total_dipole_operator")

            # take tensor product of the identity on the boson Hilbert space with this exciton operator
            _Op = np.kron(_Is, self.exciton_operator_j)

            # assign this operator to the ith position in the exciton_energy_operator
            self.exciton_dipole_operator += _Op


    def build_exciton_boson_coupling_operator(self):
        """compute the exciton-boson coupling operator in the N-exciton N'-level bosonic Hilbert space

        Arguments
        ----------
        None

        Attributes
        ----------
        exciton_boson_coupling_au : float
            interaction energy between the excitonic subsystems and the bosonic subsystem

        exciton_boson_coupling_operator_sp_b : numpy matrix
            dim x dim representation of the coupling operator proportional to b sigma^+  in the N-exciton N'-level hilbert space
            where dim is the size of the N-exciton N'-level bosonic hilbert space

        exciton_boson_coupling_operator_sm_bd : numpy matrix
            dim x dim matrix representation of the coupling operator proportional to b^+ sigma^-  in the N-exciton N'-level hilbert space
            where dim is the size of the N-exciton N'-level bosonic hilbert space


        Returns
        -------
        None

        """

        # dimension of the coupled Hilbert space
        _dim = self.exciton_boson_basis.shape[0]


        # create tensor for coupling operators
        self.exciton_boson_coupling_operator_b_sp = np.zeros((_dim, _dim))
        self.exciton_boson_coupling_operator_bd_sm = np.zeros((_dim, _dim))
        self.exciton_boson_coupling_operator = np.zeros((_dim, _dim))

        # build bosonic ladder operator
        self.build_bosonic_ladder_operators()

        # loop through excitons
        for i in range(self.number_of_excitons):

            # get the sigma_+ for the ith exciton in the N-exciton Hilbert space
            self.build_operator_for_exciton_j(i, "sigma_p", factor = self.exciton_boson_coupling_au)

            # take tensor product of the bosonic lowering operator and the sigma^+ operator times the coupling constant
            _Op = np.kron(self.b_matrix,  self.exciton_operator_j)

            # assign this operator to the ith position in the exciton_boson_coupling_operator_b_sp
            self.exciton_boson_coupling_operator_b_sp += _Op

            # get the sigma_- operator for the ith exciton in the N-exciton Hilbert space
            self.build_operator_for_exciton_j(i, "sigma_m", factor = self.exciton_boson_coupling_au)

            # take tensor product of the bosonic raising operator and the sigma^- operator times the coupling constant
            _Op = self.exciton_boson_coupling_au * np.kron(self.b_dagger_matrix,  self.exciton_operator_j)

            # assign this operator to the ith position in the exciton_boson_coupling_operator_b_sp
            self.exciton_boson_coupling_operator_bd_sm += _Op

        self.exciton_boson_coupling_operator = self.exciton_boson_coupling_operator_b_sp + self.exciton_boson_coupling_operator_bd_sm


    def compute_boson_energy_element(self, bra, ket):
        """compute the energy elements of the bosonic basis states

        Arguments
        ----------
        bra : numpy matrix
            bra state in the coupled Hilbert space

        ket : numpy matrix
            ket state in the coupled Hilbert space

        Attributes
        ----------

        boson_energy_operator : numpy matrix
            matrix representation of the bosonic energy operator in the N-exciton N'-level boson hilbert space

        Returns
        -------
        None
        """
        # This dot product will still be an array type
        E_boson_element = np.dot(bra, np.dot(self.boson_energy_operator, ket))

        return E_boson_element[0]

    def compute_exciton_energy_element(self, bra, ket):
        """compute matrix element <bra|H_QD|ket>

        Arguments
        ----------
        bra : numpy matrix
            bra state in the coupled Hilbert space

        ket : numpy matrix
            ket state in the coupled Hilbert space
        """
        
        E_exciton_element = np.dot(bra, np.dot(self.exciton_energy_operator, ket))
        return E_exciton_element[0]
    
    def compute_dipole_matrix_element(self, bra_coeffs, ket_coeffs):
        """ compute the matrix element <bra | mu | ket>

        Arguments
        ---------
        bra_coeffs : numpy array
            coefficients of the bra state 

        ket_coeffs : numpy array
            coefficients of the ket state
        
        """
        dipole_element = 0
        _dim = self.exciton_boson_basis.shape[0]

        for i in range(_dim):
            _bra = self.exciton_boson_basis[:, i]
            #print("Printing Bra Basis Vector")
            #print(_bra)
            #print("Printing bra coeff")
            #print(bra_coeffs[i])
            for j in range(_dim):
                _ket = np.array(self.exciton_boson_basis[:, i]).T #double check that this works!
                _boson_term = np.dot(_bra, np.dot(self.boson_dipole_operator, _ket))
                _exciton_term = 0.
                for k in range(self.number_of_excitons):
                    _exciton_term += np.dot(_bra, np.dot(self.exciton_dipole_operator[:,:,k], _ket))

                dipole_element += bra_coeffs[i] * ket_coeffs[j] * (_boson_term + _exciton_term)
        
                

                

            

    
    def compute_exciton_boson_coupling_element(self, bra, ket):
        """compute matrix element <bra|H_c|ket>

        Arguments
        ----------
        bra : numpy matrix
            bra state in the coupled Hilbert space

        ket : numpy matrix
            ket state in the coupled Hilbert space
        """
        E_coupling_element = 0

        for i in range(self.number_of_excitons):
            # this dot product will still be an array type
            _term = np.dot(
                bra, np.dot(self.exciton_boson_coupling_operator_b_sp[:, :, i], ket)
            )
            E_coupling_element += _term[0]
            _term = np.dot(
                bra, np.dot(self.exciton_boson_coupling_operator_bd_sm[:, :, i], ket)
            )
            E_coupling_element += _term[0]

        return E_coupling_element
    

    def compute_boson_energy_matrix(self):
        """compute the bosonic energy matrix

        Arguments
        ----------
        None

        Attributes
        ----------
        boson_energy_operator : numpy matrix
            matrix representation of the bosonic energy operator in the N-exciton N'-level boson hilbert space

        exciton_boson_basis : numpy matrix
            basis states for the collection of N excitons and the N'-level Harmonic oscillator in order
            |s> \otimes |q_1> \otimes |q_2> \otimes ... \otimes |q_N>

        boson_energy_matrix : numpy matrix
            matrix representation of the bosonic energy operator in the coupled excitonic bosonic Hilbert space

        Returns
        -------
        None
        """
        _dim = self.exciton_boson_basis.shape[0]
        #print(f" Dim is {_dim}")
        self.boson_energy_matrix = np.zeros((_dim, _dim))

        for i in range(_dim):
            for j in range(_dim):
                _bra = self.exciton_boson_basis[:, i]
                _ket = np.matrix(self.exciton_boson_basis[:, j]).T
                self.boson_energy_matrix[i, j] = self.compute_boson_energy_element(_bra, _ket)

    def compute_exciton_energy_matrix(self):
        """ compute the exciton energy matrix

        Arguments
        ----------
        None

        Attributes
        ----------
        exciton_energy_operator : numpy matrix
            dim x dim x N tensor representation of the exciton energy operator in the N-exciton hilbert

        exciton_energy_matrix : numpy matrix
            dim x dim matrix representation of the exciton energy operator in the coupled Hilbert space

        Returns
        -------
        None
        """
        _dim = self.exciton_boson_basis.shape[0]
        self.exciton_energy_matrix = np.zeros((_dim, _dim))
        for i in range(_dim):
            _bra = self.exciton_boson_basis[:, i]
            for j in range(_dim):
                _ket = np.matrix(self.exciton_boson_basis[:, j]).T
                self.exciton_energy_matrix[i, j] = self.compute_exciton_energy_element(_bra, _ket)

    def compute_exciton_boson_coupling_matrix(self):
        """ compute the exciton boson coupling

        Arguments
        ----------
        None

        Attributes
        ----------
        exciton_boson_coupling_operator : numpy matrix
            dim x dim x N tensor representation of the exciton energy operator in the N-exciton hilbert

        exciton_boson_coupling_matrix : numpy matrix
            dim x dim matrix representation of the exciton energy operator in the coupled Hilbert space

        Returns
        -------
        None
        """
        _dim = self.exciton_boson_basis.shape[0]
        self.exciton_boson_coupling_matrix = np.zeros((_dim, _dim))
        for i in range(_dim):
            _bra = self.exciton_boson_basis[:, i]
            for j in range(_dim):
                _ket = np.matrix(self.exciton_boson_basis[:, j]).T
                self.exciton_boson_coupling_matrix[i, j] = self.compute_exciton_boson_coupling_element(_bra, _ket)


    def compute_spectrum(self):
        """method that will build spin-boson Hamiltonian, diagonalize it, and return eigenvalues"""

        # build bases
        self.build_boson_basis()
        self.build_exciton_basis()
        self.build_exciton_boson_basis()

        # build operators
        self.build_exciton_boson_coupling_operator()
        self.build_exciton_energy_operator()
        self.build_boson_energy_operator()
        self.build_boson_dipole_operator()
        self.build_exciton_dipole_operator()

        # build matrices
        self.compute_boson_energy_matrix()
        self.compute_exciton_energy_matrix()
        self.compute_exciton_boson_coupling_matrix()

        # define total Hamiltonian as the sum of these matrices
        self.hamiltonian_matrix = self.boson_energy_matrix + self.exciton_energy_matrix + self.exciton_boson_coupling_matrix

        self.energy_eigenvalues, self.energy_eigenvectors = la.eigh(self.hamiltonian_matrix)
        print("Energy Eigenvalues in atomic units are")
        print(self.energy_eigenvalues)
        print("Energy eigenvalues in eV are")
        print(self.energy_eigenvalues / self.ev_to_au)

        #print(self.energy_eigenvectors[:,0])
        #print(self.energy_eigenvectors[:,1])

        #mu_01 = self.compute_dipole_matrix_element(self.energy_eigenvectors[:,0], self.energy_eigenvectors[:,1])
        #mu_02 = self.compute_dipole_matrix_element(self.energy_eigenvectors[:,0], self.energy_eigenvectors[:,2])
        #mu_03 = self.compute_dipole_matrix_element(self.energy_eigenvectors[:,0], self.energy_eigenvectors[:,3])
        #mu_12 = self.compute_dipole_matrix_element(self.energy_eigenvectors[:,1], self.energy_eigenvectors[:,2])

        #print(F'mu_01 {mu_01}')
        #print(F'mu_02 {mu_02}')
        #print(F'mu_03 {mu_03}')
        #print(F'mu_12 {mu_12}')

