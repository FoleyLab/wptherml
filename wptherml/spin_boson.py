import numpy as np
from matplotlib import pyplot as plt
from .spectrum_driver import SpectrumDriver


class SpinBosonDriver(SpectrumDriver):
    """A class for computing the dynamics and spectra of coupled exciton-boson (e.g. QD - plasmon, exciton-polariton, etc) systems using
       the spin boson for N 2-level systems coupled to an N'-level Harmonic oscillator

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

    def parse_input(self, args):
        if "number_of_excitons" in args:
            self.number_of_excitons = args["number_of_excitons"]
        else:
            self.exciton_energy = 1

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

        # convert energies from eV to au
        self.exciton_energy_au = self.exciton_energy_ev * self.ev_to_au
        self.boson_energy_au = self.boson_energy_ev * self.ev_to_au

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

        self.single_exciton_basis = np.matrix("1 0 ; 0 1")
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
        """build the boson energy operator in the N-qd N'-level coupled Hilbert space

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
        print("Printing energy operator on boson space")
        print(_energy_operator_on_boson_space)

        # build the boson energy operator in the coupled Hilbert space
        self.boson_energy_operator = np.kron(
            _energy_operator_on_boson_space, self.n_exciton_basis
        )

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

        E_boson_element = np.dot(bra, np.dot(self.boson_energy_operator, ket))

        return E_boson_element

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
        print(f" Dim is {_dim}")
        self.boson_energy_matrix = np.zeros((_dim, _dim))

        for i in range(_dim):
            for j in range(_dim):
                _bra = self.exciton_boson_basis[:, i]
                _ket = np.matrix(self.exciton_boson_basis[:, j]).T
                _element = self.compute_boson_energy_element(_bra, _ket)
                self.boson_energy_matrix[i, j] = _element[0, 0]

    def build_operator_for_exciton_j(self, j, operator="sigma_z"):
        """build operator for the j-th exciton

        Arguments
        ----------
        operator : string
            operator to build

        j : int
            index of the exciton; start couunt from 0

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
            self.single_exciton_operator = np.matrix("1 0 ; 0 -1")

        elif operator == "sigma_x":
            self.single_exciton_operator = np.matrix("0 1 ; 1 0")

        elif operator == "sigma_y":
            self.single_exciton_operator = np.matrix("0 -1j ; 1j 0")

        elif operator == "sigma_p":
            self.single_exciton_operator = np.matrix("0 1 ; 0 0")

        elif operator == "sigma_m":
            self.single_exciton_operator = np.matrix("0 0 ; 1 0")

        elif operator == "sigma_pm":
            self.single_exciton_operator = np.matrix("1 0 ; 0 0")

        else:
            # if no valid option given, use an identity
            self.single_exciton_operator = np.matrix("1 0 ; 0 1")

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

    def compute_spectrum(self):
        """method that will take values computed from spectrum_array and plot them vs wavelength"""
        spectrum_plot = np.zeros(2)  # plt.plot(self.wvlngth_variable, test_spec, 'b-')

        return spectrum_plot