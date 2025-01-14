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

    exciton_spontaneous_emission_rate_mev : float
        the spontaneous emission rate for the exciton subsystem(s) in milli eV (i.e. hbar gamma -> meV)

    exciton_spontaneous_emission_rate_au : float
        the spontaneous emission rate for the exciton subsystem(s) in atomic units

    exciton_dephasing_rate_mev : float
        the dephasing rate for the exciton subsystem(s) in milli eV (i.e. hbar gamma -> meV)

    exciton_dephasing_rate_au : float
        the dephasing rate for the exciton subsystem(s) in atomic units

    boson_spontaneous_emission_rate_mev : float
        the spontaneous emission rate for the boson subsystem in milli eV (i.e. hbar gamma -> meV)

    boson_spontaneous_emission_rate_au : float
        the spontaneous emission rate for the boson subsystem in atomic units
    
    boson_dephasing_rate_mev : float
        the dephasing rate for the boson subsystem in milli eV (i.e. hbar gamma -> meV)

    boson_dephasing_rate_au : float
        the dephasing rate for the boson subsystem in atomic units

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

        if "exciton_spontaneous_emission_rate_mev" in args:
            self.exciton_spontaneous_emission_rate_mev = args["exciton_spontaneous_emission_rate_mev"]
        else: 
            self.exciton_spontaneous_emission_rate_mev = 0.

        if "exciton_dephasing_rate_mev" in args:
            self.exciton_dephasing_rate_mev = args["exciton_dephasing_rate_mev"]
        else:
            self.exciton_dephasing_rate_mev = 0.

        if "boson_spontaneous_emission_rate_mev" in args:
            self.boson_spontaneous_emission_rate_mev = args["boson_spontaneous_emission_rate_mev"]
        else: 
            self.boson_spontaneous_emission_rate_mev = 0.

        if "boson_dephasing_rate_mev" in args:
            self.boson_dephasing_rate_mev = args["boson_dephasing_rate_mev"]
        else:
            self.boson_dephasing_rate_mev = 0.

        if "time_step_au" in args:
            self.time_step_au = args["time_step_au"]
        else:
            self.time_step_au = 0.01
        
        # convert energies from eV to au
        self.exciton_energy_au = self.exciton_energy_ev * self.ev_to_au
        self.boson_energy_au = self.boson_energy_ev * self.ev_to_au
        self.exciton_boson_coupling_au = self.exciton_boson_coupling_ev * self.ev_to_au

        # convert dissipation rates from meV to au
        self.exciton_spontaneous_emission_rate_au = self.exciton_spontaneous_emission_rate_mev * self.ev_to_au * 1e-3
        self.exciton_dephasing_rate_au = self.exciton_dephasing_rate_mev * self.ev_to_au * 1e-3
        self.boson_spontaneous_emission_rate_au = self.boson_spontaneous_emission_rate_mev * self.ev_to_au * 1e-3
        self.boson_dephasing_rate_au = self.boson_dephasing_rate_mev * self.ev_to_au * 1e-3

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
        """build operator for the j-th exciton in the coupled N-exciton hilbert space.
           THIS MUST BE LIFTED INTO THE COUPLED BOSON-EXCITON HILBERT SPACE OUTSIDE OF THIS METHOD!

        Arguments
        ----------
        operator : string
            operator to build

        j : int
            index of the exciton; start count from 0

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
            _Op = np.kron(self.b_dagger_matrix,  self.exciton_operator_j)

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

        pass 

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
    
    def build_dipole_squared_operator(self):
        """ method to build the dipole squared operator on the coupled boson-N-excition space
            currently just works for 2-spin boson system
        """

        # build mu matrix for a single exciton
        mu_matrix =  np.array([[self.exciton_ground_state_dipole_magnitude_au, self.exciton_transition_dipole_magnitude_au],
                               [self.exciton_transition_dipole_magnitude_au, self.exciton_excited_state_dipole_magnitude_au]])
        
        # build identities for spin and boson system
        _Is = np.eye(2)

        _Ib = np.eye(self.number_of_boson_levels)
        
        mu_squared = np.kron( mu_matrix @ mu_matrix, _Is) + 2 * np.kron(mu_matrix, mu_matrix) + np.kron(_Is, mu_matrix @ mu_matrix)

        self.mu_squared_operator = np.copy(mu_squared) # np.kron(_Ib, mu_squared)
    
    
    def compute_spectrum(self):
        """method that will build spin-boson Hamiltonian, diagonalize it, and return eigenvalues"""

        # build bases
        self.build_boson_basis()
        self.build_exciton_basis()
        self.build_exciton_boson_basis()

        # build operators (really matrix representations of operators)
        self.build_exciton_boson_coupling_operator()
        self.build_exciton_energy_operator()
        self.build_boson_energy_operator()
        self.build_boson_dipole_operator()
        self.build_exciton_dipole_operator()



        # define total Hamiltonian as the sum of these matrices
        self.hamiltonian_matrix = self.boson_energy_operator + self.exciton_energy_operator+ self.exciton_boson_coupling_operator

        self.energy_eigenvalues, self.energy_eigenvectors = la.eigh(self.hamiltonian_matrix)
        print("Energy Eigenvalues in atomic units are")
        print(self.energy_eigenvalues)
        print("Energy eigenvalues in eV are")
        print(self.energy_eigenvalues / self.ev_to_au)

    def build_rho_from_eigenstate(self, state_index):
        """ Method to build the density matrix from a particular energy eigenstate
        
        Parameters
        ----------
        state_index : int
            the index for the energy eigenstate

        Attributes
        ----------
        energy_eigenvectors : numpy arrays
            the energy eigenvectors

        psi : numpy array
            ket representation of the state

        rho : numpy array
            density matrix representation of the state 
        
        """

        # get the size of the ket vector
        _dim = self.exciton_boson_basis.shape[0]

        # store ket as a column vector
        self.ket = self.energy_eigenvectors[:,state_index].reshape(_dim, 1)
        
        # store bra as a row vector
        self.bra = self.ket.T.conj()

        # compute density matrix as |state><state|
        self.rho = self.ket @ self.bra 


    def build_rho_from_ket(self, ket):
        """ Method to build the density matrix from a particular ket state
        
        Arguments
        ---------
        ket : numpy array
            the ket state

        Attributes
        ----------
        psi : numpy array
            ket representation of the state

        psi_star : numpy array
            bra representation of the state

        rho : numpy array

        Returns
        -------
        None
        """
        # store ket as a column vector to self.psi
        self.psi = ket

        # store bra as a row vector to self.psi_star
        self.psi_star = self.psi.T.conj()

        # compute density matrix as |state><state|
        self.rho = self.psi @ self.psi_star

    def kron_index_map_AB(self,A, B):
        """ Creates a map between the row and column of the Kronecker product of two arrays
            and the composite indices corresponding to the original arrays.

        Arguments
        ----------
        A : numpy array
            first array

        B : numpy array
            second array

        Returns
        -------
        index_map : dictionary
            dictionary with keys as the row and column indices and values as the composite indices 
        """
        rowsA, colsA = A.shape
        rowsB, colsB = B.shape

        index_map = {}
        for rowAB in range(rowsA * rowsB):
            rowA = rowAB // rowsB
            rowB = rowAB % rowsB
            for colAB in range(colsA * colsB):
                colA = colAB // colsB
                colB = colAB % colsB
                index_map[(rowAB, colAB)] = (rowA, colA, rowB, colB)

        return index_map
    

    def kron_index_map_ABC(self, A, B, C):
        """ Creates a map between the row and column of the Kronecker product of three arrays
            and the composite indices corresponding to the original arrays.

        Arguments
        ----------
        A : numpy array
            first array
        B : numpy array
            second array
        C : numpy array
            third array

        Returns
        -------
        index_map : dictionary
            dictionary with keys as the row and column indices and values as the composite indices 
        """
        rowsA, colsA = A.shape
        rowsB, colsB = B.shape
        rowsC, colsC = C.shape

        print("Rows and Cols of A, B, C")
        print(rowsA, colsA)
        print(rowsB, colsB)
        print(rowsC, colsC)
        index_map = {}
        for rowABC in range(rowsA * rowsB * rowsC):
            for colABC in range(colsA * colsB * colsC):
                i = rowABC // (rowsB * rowsC)
                temp = rowABC % (rowsB * rowsC)
                k = temp // rowsC
                m = temp % rowsC

                j = colABC // (colsB * colsC)
                temp = colABC % (colsB * colsC)
                l = temp // colsC
                n = temp % colsC

                index_map[(rowABC, colABC)] = (i, j, k, l, m, n)

        return index_map


    def compute_partial_traces(self, rho):
        """ Method to compute all possible single partial traces of a composite density matrix; currently
            supports a caivty x spin1 system or a cavity x spin 1 x spin 2 system  
            The order of the subsystems is assumed to be |cavity> x |spin1> x |spin2>
        
        Arguments
        ---------
        rho : numpy array
            the density matrix

        Attributes
        ----------
        rdm_cavity : numpy array
            reduced density matrix for the cavity

        rdm_spin1 : numpy array 
            reduced density matrix for the first spin

        rdm_spin2 : numpy array (if applicable)
            reduced density matrix for the second spin
        """
        # right now only support partial trace when we have 1 or 2 spin systems and 1 boson system
        # we will consider the single spin system case first
        if self.number_of_excitons == 1:
            # get the index map for the Kronecker product of the boson and single exciton basis
            _index_map = self.kron_index_map_AB(self.boson_basis, self.single_exciton_basis)

            self.rdm_cavity = np.zeros_like(self.boson_basis)
            self.rdm_spin1 = np.zeros_like(self.single_exciton_basis)

            # take both partial traces
            for (_row, _col), (_i, _j, _k, _l) in _index_map.items():
                if _k==_l:
                    self.rdm_cavity[_i, _j] += rho[_row, _col]
                if _i==_j:
                    self.rdm_spin1[_k, _l] += rho[_row, _col]   

        elif self.number_of_excitons == 2:
            # get the index map for the Kronecker product of the boson and two exciton basis
            _index_map = self.kron_index_map_ABC(self.boson_basis, self.single_exciton_basis, self.single_exciton_basis)

            # get dimensions of three individual sub-matrices
            _dimA = self.boson_basis.shape[0]
            _dimB = self.single_exciton_basis.shape[0]
            _dimC = self.single_exciton_basis.shape[0]

            # initialize the three possible 2-rdms
            self.rdm_cavity_spin1 = np.zeros((_dimA * _dimB, _dimA * _dimB))
            self.rdm_cavity_spin2 = np.zeros((_dimA * _dimC, _dimA * _dimC))
            self.rdm_spin1_spin2 = np.zeros((_dimB * _dimC, _dimB * _dimC))

            # take all three partial traces
            for (_row, _col), (_i, _j, _k, _l, _m, _n) in _index_map.items():

                # partial trace over spin 2
                if _m == _n:
                    _trow = _i * _dimB + _k
                    _tcol = _j * _dimB + _l
                    self.rdm_cavity_spin1[_trow, _tcol] += rho[_row, _col]

                # partial trace over spin 1
                if _k == _l:
                    _trow = _i * _dimC + _m
                    _tcol = _j * _dimC + _n
                    self.rdm_cavity_spin2[_trow, _tcol] += rho[_row, _col]

                # partial trace over cavity
                if _i == _j:
                    _trow = _k * _dimC + _m
                    _tcol = _l * _dimC + _n
                    self.rdm_spin1_spin2[_trow, _tcol] += rho[_row, _col]

    def compute_entanglement_entropy(self, rdm):
        """ Method to compute the entanglement entropy of a reduced density matrix

        Arguments
        ---------
        rdm : numpy array
            the reduced density matrix

        Returns
        -------
        entropy : float
            the entanglement entropy
        """
        _eigenvalues = la.eigvalsh(rdm)
        _entropy = 0
        for _eig in _eigenvalues:
            if _eig > 0:
                _entropy -= _eig * np.log(_eig)
        return _entropy
                

    
    def compute_lindblad_exciton_i_on_rho(self, i):
        """
        Compute the Lindblad superoperator contribution for a specific spin on the density matrix.

        This method calculates the contribution to the time derivative of the density matrix (`rho`) 
        due to the Lindblad superoperator associated with spin `i`. It includes terms for spontaneous 
        emission and dephasing processes in the exciton system.

        Parameters
        ----------
        i : int
            Index of the spin (exciton) for which the Lindblad superoperator is computed.

        Returns
        -------
        np.ndarray
            The time derivative of the density matrix (`rho`) contributed by the Lindblad superoperator 
            for spin `i`. The result is an array of the same shape as the density matrix.

        Notes
        -----
        The Lindblad terms are constructed as:
        
        - Spontaneous emission term:
        \[
        T_{se} = \sigma^+ \sigma^- \rho + \rho \sigma^+ \sigma^- - 2 \sigma^- \rho \sigma^+
        \]
        
        - Dephasing term:
        \[
        T_d = \sigma^+ \sigma^- \rho + \rho \sigma^+ \sigma^- - 2 \sigma^+ \sigma^- \rho \sigma^+ \sigma^-
        \]
        
        The final time derivative of the density matrix is given by:
        \[
        \dot{\rho} = -\frac{\gamma_p}{2} T_{se} - \gamma_d T_d
        \]
        where \(\gamma_p\) is the spontaneous emission rate and \(\gamma_d\) is the dephasing rate.

        The operators \(\sigma^+ \sigma^-\), \(\sigma^+\), and \(\sigma^-\) are constructed for the specified 
        spin `i` and are extended to the full Hilbert space via a tensor product with the bosonic identity.

        """

        # get the gamma factors
        _gamma_p = self.exciton_spontaneous_emission_rate_au
        _gamma_d = self.exciton_dephasing_rate_au

        # copy current density matrix
        _rho_t = np.copy(self.rho)
        
        # build identity on the boson space
        _Is = np.copy(self.boson_basis)

        # get sigma^+ sigma^- operator for spin i
        self.build_operator_for_exciton_j(i, "sigma_pm", 1.)

        # take tensor product of boson identity with this exciton operator
        _sigma_pm = np.kron(_Is, self.exciton_operator_j)

        # get sigma^+ operator for spin i
        self.build_operator_for_exciton_j(i, "sigma_p", 1.)

        # take tensor product of boson identity with this exciton operator
        _sigma_p = np.kron(_Is, self.exciton_operator_j)

        # get sigma^- operator for spin i
        self.build_operator_for_exciton_j(i, "sigma_m", 1.)

        # take tensor product of boson identity with this exciton operator
        _sigma_m = np.kron(_Is, self.exciton_operator_j)

        # compute spontaneous emission term
        _T_se = _sigma_pm @ _rho_t + _rho_t @ _sigma_pm - 2 * _sigma_m @ _rho_t @ _sigma_p 

        # compute dephasing term
        _T_d = _sigma_pm @ _rho_t + _rho_t @  _sigma_pm - 2 * _sigma_pm @ _rho_t @ _sigma_pm 

        # take total time derivative of rho coming from Lindblad term for exciton i
        _rho_dot = -_gamma_p / 2 * _T_se - _gamma_d * _T_d

        return _rho_dot
    
    def compute_lindblad_boson_on_rho(self):
        """
        Compute the Lindblad superoperator contribution for the boson on the density matrix.

        This method calculates the contribution to the time derivative of the density matrix (`rho`) 
        due to the Lindblad superoperator associated with the bosonic subsystem. It includes terms for spontaneous 
        emission and dephasing processes in the exciton system.

        Parameters
        ----------
        i : int
            Index of the spin (exciton) for which the Lindblad superoperator is computed.

        Returns
        -------
        np.ndarray
            The time derivative of the density matrix (`rho`) contributed by the Lindblad superoperator 
            for spin `i`. The result is an array of the same shape as the density matrix.

        Notes
        -----
        The Lindblad terms are constructed as:
        
        - Spontaneous emission term:
        \[
        T_{se} = b^+ b \rho + \rho b^+ b - 2 b \rho b^+
        \]
        
        - Dephasing term:
        \[
        T_d = b^+ b \rho + \rho b^+ b - 2 b^+ b \rho b^+ b
        \]
        
        The final time derivative of the density matrix is given by:
        \[
        \dot{\rho} = -\frac{\gamma_p}{2} T_{se} - \gamma_d T_d
        \]
        where \(\gamma_p\) is the spontaneous emission rate and \(\gamma_d\) is the dephasing rate.

        The operators \(b^+ b \), \(b^+ \), and \(b \) are constructed for the boson system
        and are extended to the full Hilbert space via a tensor product with the identity for N spins.

        """

        # get the gamma factors
        _gamma_p = self.boson_spontaneous_emission_rate_au
        _gamma_d = self.boson_dephasing_rate_au

        # copy current density matrix
        _rho_t = np.copy(self.rho)
        
        # build identity on the boson space
        _Is = np.eye(self.exciton_basis_dimension)

        # get b^+ and b operators tensor products with exciton identity
        _bd = np.kron(self.b_dagger_matrix, _Is)
        _b = np.kron(self.b_matrix, _Is)

        # compute spontaneous emission term
        _T_se = _bd @ _b @ _rho_t + _rho_t @ _bd @ _b - 2 * _b @ _rho_t @ _bd

        # compute dephasing term
        _T_d = _bd @ _b @ _rho_t + _rho_t @  _bd @ _b - 2 * _bd @ _b @ _rho_t @ _bd @ _b

        # take total time derivative of rho coming from Lindblad term for exciton i
        _rho_dot = -_gamma_p / 2 * _T_se - _gamma_d * _T_d

        return _rho_dot
    
    def compute_hamiltonian_on_rho(self):
        """
        Compute the time derivative of the density matrix from the Hamiltonian commutator.

        This method calculates the unitary contribution to the time evolution of the density 
        matrix (`rho`) based on the commutator with the system Hamiltonian.

        The evolution is given by:
        \[
        \dot{\rho} = -i [H, \rho]
        \]
        where \([H, \rho] = H \rho - \rho H\) is the commutator of the Hamiltonian (`H`) 
        and the density matrix (`rho`).

        Returns
        -------
        np.ndarray
            The time derivative of the density matrix (`rho_dot`) as an array of the same shape.

        Notes
        -----
        - The Hamiltonian matrix (`H`) is assumed to be Hermitian, as required in quantum mechanics.
        - The computation uses the imaginary unit \(i = \sqrt{-1}\), represented as `1j` in Python.

        Examples
        --------
        Assuming `self.hamiltonian_matrix` and `self.rho` are properly initialized:
        
        >>> rho_dot = obj.compute_hamiltonian_on_rho()
        >>> print(rho_dot)
        [[...]]
        """
        # imaginary unit
        ci = 0+1j

        # copy density matrix
        _rho_t = np.copy(self.rho)

        # copy Hamiltonian
        _H = np.copy(self.hamiltonian_matrix)

        # compute commutator
        _rho_dot = -ci * ( _H @ _rho_t - _rho_t @ _H )

        return _rho_dot
    

    def compute_time_derivative_of_rho(self):
        """
        Compute the total time derivative of the density matrix.

        This method calculates the time derivative of the density matrix (`rho`) 
        by combining the unitary evolution due to the system Hamiltonian and 
        the non-unitary evolution from Lindblad operators, which account for 
        dissipation and decoherence.

        The total time derivative is given by:
        \[
        \dot{\rho} = -i [H, \rho] + \mathcal{L}_{\text{boson}}[\rho] + \sum_i \mathcal{L}_{\text{exciton}, i}[\rho]
        \]
        where:
        - \([H, \rho]\) is the commutator capturing unitary evolution.
        - \(\mathcal{L}_{\text{boson}}[\rho]\) is the Lindblad contribution from the bosonic mode.
        - \(\mathcal{L}_{\text{exciton}, i}[\rho]\) is the Lindblad contribution from the \(i\)-th exciton.

        Returns
        -------
        np.ndarray
            The time derivative of the density matrix (`rho_dot`) as an array 
            of the same shape.

        Notes
        -----
        - The method uses `compute_hamiltonian_on_rho` for the unitary evolution.
        - The non-unitary evolution contributions are calculated by `compute_lindblad_boson_on_rho` 
        and `compute_lindblad_exciton_i_on_rho` for each exciton.
        - The computed time derivative is stored in `self.rho_dot` for later use.

        Examples
        --------
        Assuming `self.hamiltonian_matrix`, `self.rho`, and relevant Lindblad rates are initialized:

        >>> rho_dot = obj.compute_time_derivative_of_rho()
        >>> print(rho_dot)
        [[...]]
        """

        # compute the unitary contributions
        _rho_dot = self.compute_hamiltonian_on_rho()

        # compute non-unitary contribution from boson
        _rho_dot += self.compute_lindblad_boson_on_rho()

        # for each exciton, compute non-unitary contribution from each excition
        for i in range(self.number_of_excitons):
            _rho_dot += self.compute_lindblad_exciton_i_on_rho(i)

        self.rho_dot = np.copy(_rho_dot)
        return _rho_dot
    

    def rk4_update_on_rho(self):
        """
        Performs RK4 update on attribute self.rho which is the density matrix

        Arguments
        ---------
        None

        Attributes
        ----------
        self.rho : numpy array
            the density matrix

        self.time_step_au : float
            the time step in atomic units

        Return
        ------
        None
        """
        # get time step variable
        _h = self.time_step_au

        # copy current rho
        _rho_tn = np.copy(self.rho)

        # get rate for first Euler update
        _k1 = self.compute_time_derivative_of_rho()

        # perform half update
        _rho_u1 = _rho_tn + _h/2 * _k1

        # copy half updated rho to self.rho
        self.rho = np.copy(_rho_u1)

        # get rate for second Euler update
        _k2 = self.compute_time_derivative_of_rho()

        # perform half update
        _rho_u2 = _rho_tn + _h/2 * _k2

        # copy half updated rho to self.rho
        self.rho = np.copy(_rho_u2)

        # get rate for third Euler update
        _k3 = self.compute_time_derivative_of_rho()

        # perform full update
        _rho_u3 = _rho_tn + _h * _k3

        # copy updated rho to self.rho
        self.rho = np.copy(_rho_u3)

        # get rate for fourth Euler update
        _k4 = self.compute_time_derivative_of_rho()

        # perform full update
        _rho_update = _rho_tn + _h/6 * (_k1 + 2*_k2 + 2*_k3 + _k4)

        self.rho = np.copy(_rho_update)
        
        return _rho_update






        





