import numpy as np
from scipy.interpolate import UnivariateSpline


class Therml:
    """Collects methods for the computation of thermal radiative figures of merit

    Attributes
    ----------
    self.temperture : float
        temperature of the structure in Kelvin

    self.atmospheric_temperature : float
        temperature of the atmosphere in Kelvin

    self.lambda_bandgap : float
        bandgap wavelength of the target PV for PV or STPV in meters

    self.solar_angle : float
        angle of the incident solar spectrum in radians

    self.blackbody_spectrum : numpy array of floats
        Planck's blackbody spectrum for a given temperature

    self.thermal_emission_array : numpy array of floats
        thermal emission spectrum of structure for a given temperature

    self.blackbody_power_density : float
        total power density radiated into a hemisphere of a blackbody at a given temperature

    self.power_density : float
        total power density radiated by structure at a given temperature

    self.power_density_gradient : numpy array of floats (will be computed by this function)
        the gradient vector related to the total power density wrt changes in thicknesses of each layer

    self.photopic_luminosity_array : numpy array of floats
        the array of photopic luminosity values corresponding to each value of wavelength_array

    self.stpv_power_density : float (will be computed by this function)
        useful (sub-bandgap) power density radiated into a hemisphere

    self.stpv_power_density_gradient : numpy array of floats (will be computed by this function)
        the gradient vector related to the stpv power density wrt changes in thicknesses of each layer

    self.stpv_spectral_efficiency : float (will be computed by this function)
        spectral efficiency of an stpv emitter

    self.stpv_spectral_efficiency_gradient : numpy array of floats (will be computed by this function)
        the gradient vector related to the spectral efficiency wrt changes in thicknesses of each layer


    Returns
    -------
        None

    """

    def __init__(self, args):
        """constructor for the Therml class"""
        # parse args
        self._parse_therml_input(args)
        # self._compute_therml_spectrum()
        # self._compute_power_density()

    def _parse_therml_input(self, args):
        """method to parse the user inputs and define structures / simulation

        Returns
        -------
        None

        """
        if "temperature" in args:
            # user input expected in kelvin
            self.temperature = args["temperature"]

        else:
            # default temperature is 300 K
            self.temperature = 300

        if "bandgap wavelength" in args:
            self.lambda_bandgap = args["bandgap wavelength"]
        else:
            # default is ~InGaAsSb bandgap
            self.lambda_bandgap = 2254e-9

        if "atmospheric temperature" in args:
            self.atmospheric_temperature = args["atmospheric temperature"]
        else:
            self.atmospheric_temperature = 300

        if "solar angle" in args:
            self.solar_angle = args["solar angle"]
            self.solar_angle *= np.pi / 180
        else:
            self.solar_angle = 30 * np.pi / 180

    def _compute_therml_spectrum(self, wavelength_array, emissivity_array):
        """method to compute thermal emission spectrum of a structure

        Arguments
        ---------
        wavelength_array : numpy array of floats
            the array of wavelengths across which thermal emission spectrum will be computed

        emissivity_array : numpy array of floats
            the array of emissivity spectrum for the structure that you will compute the thermal emission of

        Attributes
        ----------
        blackbody_spectrum : numpy array of floats
            Planck's blackbody spectrum for a given temperature

        thermal_emission_array : numpy array of floats
            thermal emission spectrum of structure for a given temperature

        References
        ----------
        blackbody spectrum : Eq. (13) of https://github.com/FoleyLab/wptherml/blob/master/docs/Equations.pdf

        thermal_emission_array : Eq (12) of https://github.com/FoleyLab/wptherml/blob/master/docs/Equations.pdf
        with $\theta=0$
        """

        self.blackbody_spectrum = self._compute_blackbody_spectrum(
            wavelength_array, self.temperature
        )
        self.thermal_emission_array = self.blackbody_spectrum * emissivity_array

    def _compute_therml_spectrum_gradient(
        self, wavelength_array, emissivity_gradient_array
    ):
        """method to compute thermal emission spectrum of a structure

        Arguments
        ---------
        wavelength_array : numpy array of floats
            the array of wavelengths across which thermal emission spectrum will be computed

        emissivity_array : numpy array of floats
            the array of emissivity spectrum for the structure that you will compute the thermal emission of

        Attributes
        ----------
        blackbody_spectrum : numpy array of floats
            Planck's blackbody spectrum for a given temperature

        thermal_emission_array : numpy array of floats
            thermal emission spectrum of structure for a given temperature

        References
        ----------
        blackbody spectrum : Eq. (13) of https://github.com/FoleyLab/wptherml/blob/master/docs/Equations.pdf

        thermal_emission_array : Eq (12) of https://github.com/FoleyLab/wptherml/blob/master/docs/Equations.pdf
        with $\theta=0$
        """
        # get the dimension of the gradient vector
        _ngr = len(emissivity_gradient_array[0, :])
        # get the number of wavelengths
        _nwl = len(wavelength_array)
        # initialize the gradient array
        self.thermal_emission_gradient_array = np.zeros((_nwl, _ngr))

        self.blackbody_spectrum = self._compute_blackbody_spectrum(
            wavelength_array, self.temperature
        )

        for i in range(0, _ngr):
            self.thermal_emission_gradient_array[:, i] = (
                self.blackbody_spectrum * emissivity_gradient_array[:, i]
            )

    def _compute_blackbody_spectrum(self, wavelength_array, T):
        # speed of light in SI
        c = 299792458
        # plancks constant in SI
        h = 6.62607004e-34
        # boltzmanns constant in SI
        kb = 1.38064852e-23

        _bb_spectrum = 2 * h * c**2 / wavelength_array**5
        _bb_spectrum /= np.exp(h * c / (wavelength_array * kb * T)) - 1
        return _bb_spectrum

    def _compute_pv_stpv_power_density(self, wavelength_array):
        """method to compute the radiated power density of a PV-STPV structure specifically
        in the 0.3 - 0.5 eV range (~2450-4150 nm range)

        """
        # set lower- and upper limits on wavelength for integration
        _lambda_min = 2450e-9
        _lambda_max = 4150e-9

        # get the index associated with these upper- and lower-wavelengths
        _min_idx = np.abs(wavelength_array - _lambda_min).argmin()
        _max_idx = np.abs(wavelength_array - _lambda_max).argmin()

        # integrate the thermal emission spectrum over wavelength range using np.trapz
        self.pv_stpv_exciton_splitting_power = np.pi * np.trapz(
            self.thermal_emission_array[_min_idx:_max_idx],
            wavelength_array[_min_idx:_max_idx],
        )

    def _compute_power_density(self, wavelength_array):
        """method to compute the power density from blackbody spectrum and thermal emission spectrum

        Attributes
        ----------
        self.blackbody_power_density : float
            total power density radiated into a hemisphere of a blackbody at a given temperature

        self.power_density : float
            total power density radiated by structure at a given temperature

        References
        ----------
        Equation (15) and (16) of https://github.com/FoleyLab/wptherml/blob/master/docs/Equations.pdf

        """

        # integrate blackbody spectrum over wavelength using np.trapz
        self.blackbody_power_density = np.trapz(
            self.blackbody_spectrum, wavelength_array
        )

        # integrate the thermal emission spectrum over wavelength using np.trapz
        self.power_density = np.trapz(self.thermal_emission_array, wavelength_array)

        # account for angular integrals over hemisphere (assuming no angle dependence of emissivity)
        self.blackbody_power_density *= np.pi
        self.power_density *= np.pi

        # compute Blackbody power density from Stefan-Boltzmann law for validation
        # Stefan-Boltzmann constant
        sig = 5.670374419e-8
        self.stefan_boltzmann_law = sig * self.temperature**4

    def _compute_power_density_gradient(self, wavelength_array):
        """method to compute the gradient of the power density of a thermal emitter

        Arguments
        ---------
        wavelength_array : numpy array of floats
            the wavelengths over which the power density spectrum has been computed

        Attributes
        ----------

        self.power_density_gradient : numpy array of floats (will be computed by this function)
            the gradient vector related to the total power density wrt changes in thicknesses of each layer

        References
        ----------
        Equation (5) of https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.013018

        """
        _ngr = len(self.thermal_emission_gradient_array[0, :])
        # integrate the thermal emission spectrum over wavelength using np.trapz
        self.power_density_gradient = np.zeros(_ngr)

        for i in range(0, _ngr):
            self.power_density_gradient[i] = np.pi * np.trapz(
                self.thermal_emission_gradient_array[:, i], wavelength_array
            )

    def _compute_photopic_luminosity(self, wavelength_array):
        """computes the photopic luminosity function from a Gaussian fit

        Arguments
        ---------
        wavelength_array : numpy array of floats
            the array of wavelengths over which you will compute the photopic luminosity

        Attributes
        ----------
        self.photopic_luminosity_array : numpy array of floats
            the array of photopic luminosity values corresponding to each value of wavelength_array

        References
        ----------
        Data taken from http://www.cvrl.org/database/data/lum/linCIE2008v2e_5.htm
        """
        a = 1.02433
        b = 2.59462e14
        c = 5.60186e-07

        self._photopic_luminosity_array = a * np.exp(-b * (wavelength_array - c) ** 2)

    def _compute_stpv_power_density(self, wavelength_array):
        """method to compute the stpv power density from the thermal emission spectrum of a structure

        Arguments
        ---------
        wavelength_array : numpy array of floats
            the wavelengths over which the thermal emission spectrum is known

        Attributes
        ----------

        thermal_emission_array : numpy array of floats (already assigned)
            the thermal emission spectrum for each value of wavelength array for the stpv structure

        lambda_bandgap : float (already assigned)
            the bandgap wavelength -> upper limit on the integral for the stpv_power_density

        stpv_power_density : float (will be computed by this function)
            useful (sub-bandgap) power density radiated into a hemisphere


        References
        ----------
        Equation (17) of https://github.com/FoleyLab/wptherml/blob/master/docs/Equations.pdf

        """
        # compute the useful power density spectrum
        power_density_array = (
            self.thermal_emission_array * wavelength_array
        ) / self.lambda_bandgap

        # determine the index corresponding to lambda_bandgap in the wavelength_array
        # which will be used to determine the appropriate slice to feed to np.trapz
        bg_idx = np.abs(wavelength_array - self.lambda_bandgap).argmin()

        # integrate the power density between 0 to lambda_bandgap
        # by feeding the slice of the power_density_array and wavelength_array
        # from 0:bg_idx to the trapz function
        self.stpv_power_density = np.pi * np.trapz(
            power_density_array[:bg_idx], wavelength_array[:bg_idx]
        )

    def _compute_stpv_power_density_gradient(self, wavelength_array):
        """method to compute the power density from blackbody spectrum and thermal emission spectrum
        Arguments
        ---------
        wavelength_array : numpy array of floats
            the wavelengths over which the power density spectrum has been computed

        Attributes
        ----------
        self.stpv_power_density_gradient : numpy array of floats (will be computed by this function)
            the gradient vector related to the stpv power density wrt changes in thicknesses of each layer

        References
        ----------
        Equation (5) of https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.013018

        """
        _ngr = len(self.thermal_emission_gradient_array[0, :])
        # integrate the thermal emission spectrum over wavelength using np.trapz
        self.stpv_power_density_gradient = np.zeros(_ngr)
        # compute the useful power density spectrum

        for i in range(0, _ngr):
            stpv_power_density_array_prime = (
                self.thermal_emission_gradient_array[:, i]
                * wavelength_array
                / self.lambda_bandgap
            )
            self.stpv_power_density_gradient[i] = np.pi * np.trapz(
                stpv_power_density_array_prime, wavelength_array
            )

    def _compute_stpv_spectral_efficiency(self, wavelength_array):
        """method to compute the stpv spectral efficiency from the thermal emission spectrum of a structure

        Arguments
        ---------
        wavelength_array : numpy array of floats
            the wavelengths over which the thermal emission spectrum is known

        Attributes
        ----------

        thermal_emission_array : numpy array of floats (already assigned)
            the thermal emission spectrum for each value of wavelength array for the stpv structure

        lambda_bandgap : float (already assigned)
            the bandgap wavelength -> upper limit on the integral for the stpv_power_density

        stpv_power_density : float (will be computed by this function)
            useful (sub-bandgap) power density radiated into a hemisphere

        power_density : float (will be computed by this function)
            total power density radiated into a hemisphere

        stpv_spectral_efficiency : float (will be computed by this function)
            spectral efficiency of an stpv emitter

        References
        ----------
        Equation (18) of https://github.com/FoleyLab/wptherml/blob/master/docs/Equations.pdf

        """

        self._compute_stpv_power_density(wavelength_array)
        self._compute_power_density(wavelength_array)
        self.stpv_spectral_efficiency = self.stpv_power_density / self.power_density

    def _compute_stpv_spectral_efficiency_gradient(self, wavelength_array):
        """method to compute the gradient of the
           stpv spectral efficiency from the thermal emission spectrum of a structure

        Arguments
        ---------
        wavelength_array : numpy array of floats
            the wavelengths over which the thermal emission spectrum is known

        Attributes
        ----------
        self.stpv_spectral_efficiency_gradient : numpy array of floats (will be computed by this function)
            the gradient vector related to the spectral efficiency wrt changes in thicknesses of each layer

        References
        ----------
        Equation (4) of https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.013018

        """
        # get the number of elements in the gradient
        _ngr = len(self.thermal_emission_gradient_array[0, :])

        # initialize the gradient array
        self.stpv_spectral_efficiency_gradient = np.zeros(_ngr)

        # using the notation from Eq. (4)
        # from https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.013018
        self._compute_stpv_power_density(wavelength_array)
        self._compute_power_density(wavelength_array)

        _P = self.power_density
        _rho = self.stpv_power_density

        # determine the index corresponding to lambda_bandgap in the wavelength_array
        # which will be used to determine the appropriate slice to feed to np.trapz
        _bg_idx = np.abs(wavelength_array - self.lambda_bandgap).argmin()

        for i in range(0, _ngr):
            _rho_prime_integrand = (
                self.thermal_emission_gradient_array[:_bg_idx, i]
                * wavelength_array[:_bg_idx]
                / self.lambda_bandgap
            )
            _rho_prime = np.pi * np.trapz(
                _rho_prime_integrand, wavelength_array[:_bg_idx]
            )
            _P_prime = np.pi * np.trapz(
                self.thermal_emission_gradient_array[:, i], wavelength_array
            )
            self.stpv_spectral_efficiency_gradient[i] = (
                _rho_prime * _P - _P_prime * _rho
            ) / (_P * _P)

    def _compute_pv_short_circuit_current(
        self, wavelength_array, absorptivity_array, spectral_response, solar_spectrum
    ):
        """method to approximate the short circuit current of a PV cell

        Arguments
        ---------
        wavelength_array : numpy array of floats
            the wavelengths over which the absorptivity of the structure is known

        absorptivity_array : numpy array of floats
            the absorptivity / emissivity spectrum of the PV structure

        spectral_response : numpy array of floats
            the mapping between power in  (W) / current out (Amps) of the active layer

        solar_spectrum : numpy array of floats
            the solar spectrum incident on the PV structure

        Attributes
        ----------
        self.short_circuit_current : float
            the short circuit current in Amps / m^2 of the surface

        Returns
        -------
        None

        """
        self.pv_short_circuit_current = np.trapz(
            absorptivity_array * spectral_response * solar_spectrum, wavelength_array
        )

    def _compute_luminous_efficiency(self, wavelength_array):
        """method to compute the luminous efficiency for an incandescent from the thermal emission spectrum of a structure

        Arguments
        ---------
        wavelength_array : numpy array of floats
            the wavelengths over which the thermal emission spectrum is known

        Attributes
        ----------

        self.thermal_emission_array : numpy array of floats (already assigned)
            the thermal emission spectrum for each value of wavelength array for the stpv structure

        self.photopic_luminosity_array : numpy array of floats (can be computed by calling self._compute_photopic_luminosity(wavelength_array))
            photopic luminosity function values corresponding to wavelength_array

        self.luminous_efficiency : float (will be computed by this function)
            the spectral efficiency of the incandescent source

        References
        ----------
        Equation (27) of https://github.com/FoleyLab/wptherml/blob/master/docs/Equations.pdf

        """
        # self._compute_therml_spectrum(wavelength_array, emissivity_array)
        self._compute_photopic_luminosity(wavelength_array)
        vl = self._photopic_luminosity_array
        TE = self.thermal_emission_array

        Numerator = np.trapz(vl * TE, wavelength_array)
        Denominator = np.trapz(TE, wavelength_array)

        self.luminous_efficiency = Numerator / Denominator

    def _compute_thermal_radiated_power(
        self,
        emissivity_array_s,
        emissivity_array_p,
        theta_vals,
        theta_weights,
        wavelength_array,
    ):
        """Method to compute the power radiated by the cooling structure

        Attributes
        ----------
        None

        Returns
        -------
        P_rad

        References
        ----------
        See Eq. (2) of https://www.nature.com/articles/nature13883
        """

        num_angles = len(theta_vals)
        self._compute_therml_spectrum(wavelength_array, emissivity_array_s[0, :])

        # loop over angles
        P_rad = 0.0
        for i in range(0, num_angles):
            _TE = (
                self.blackbody_spectrum
                * np.cos(theta_vals[i])
                * 0.5
                * (emissivity_array_p[i, :] + emissivity_array_s[i, :])
            )
            _TE_INT = np.trapz(_TE, wavelength_array)
            P_rad += _TE_INT * np.sin(theta_vals[i]) * theta_weights[i]

        P_rad *= np.pi * 2

        return P_rad

    def _compute_thermal_radiated_power_gradient(
        self,
        emissivity_gradient_array_s,
        emissivity_gradient_array_p,
        theta_vals,
        theta_weights,
        wavelength_array,
    ):
        """Method to compute the gradient of the power radiated by the cooling structure

        Attributes
        ----------
        None

        Returns
        -------
        _emitted_thermal_spectrum_gradient

        References
        ----------
        See Eq. (2) of https://www.nature.com/articles/nature13883

        """
        num_angles = len(theta_vals)
        # we don't care about the emissivity - just calling this for the blackbody spectrum
        self._compute_therml_spectrum(
            wavelength_array, emissivity_gradient_array_p[0, :, 0]
        )

        _ngr = len(emissivity_gradient_array_s[0, 0, :])
        _nth = len(emissivity_gradient_array_s[:, 0, 0])

        # instantiate P_rad_prime
        _emitted_thermal_spectrum_gradient = np.zeros(_ngr)
        for i in range(0, _ngr):
            _P_rad_prime = 0
            for j in range(0, _nth):
                _TE = (
                    self.blackbody_spectrum
                    * np.cos(theta_vals[j])
                    * 0.5
                    * (
                        emissivity_gradient_array_p[j, :, i]
                        + emissivity_gradient_array_s[j, :, i]
                    )
                )
                _TE_INT = np.trapz(_TE, wavelength_array)

                _P_rad_prime += (
                    2 * np.pi * _TE_INT * np.sin(theta_vals[j]) * theta_weights[j]
                )
            _emitted_thermal_spectrum_gradient[i] = _P_rad_prime

        return _emitted_thermal_spectrum_gradient

    def _compute_atmospheric_radiated_power(
        self,
        atmospheric_transmissivity,
        emissivity_array_s,
        emissivity_array_p,
        theta_vals,
        theta_weights,
        wavelength_array,
    ):
        """Method to compute the power radiated by the atmosphere and absorbed by cooling structure

        Attributes
        ----------
        None

        Returns
        -------
        P_atm

        References
        ----------
        See Eq. (3) of https://www.nature.com/articles/nature13883

        """
        num_angles = len(theta_vals)

        # make sure we are getting the blackbody spectrum of the atmosphere
        # store the structure temperature
        _T_temp = self.temperature
        # update the structure temperature, which is the attribute that
        # is used for computing the blackbody spectrum in _compute_therml_spectrum
        self.temperature = self.atmospheric_temperature
        self._compute_therml_spectrum(wavelength_array, emissivity_array_s[0, :])
        # set the structure temperature back to _T_temp in case
        # one wants to compute the thermal emission of the structure again!
        self.temperature = _T_temp
        P_atm = 0.0
        for i in range(0, num_angles):
            # get the term that goes in the exponent of the atmospheric transmissivity
            _o_over_cos_t = 1 / np.cos(theta_vals[i])
            _emissivity_atm = (
                np.ones(len(atmospheric_transmissivity))
                - atmospheric_transmissivity**_o_over_cos_t
            )
            _TE_atm = self.blackbody_spectrum * _emissivity_atm * np.cos(theta_vals[i])
            _absorbed_TE_spectrum = (
                _TE_atm * 0.5 * (emissivity_array_p[i, :] + emissivity_array_s[i, :])
            )
            _absorbed_TE = np.trapz(_absorbed_TE_spectrum, wavelength_array)
            P_atm += _absorbed_TE * np.sin(theta_vals[i]) * theta_weights[i]
        P_atm *= 2 * np.pi

        return P_atm

    def _compute_atmospheric_radiated_power_gradient(
        self,
        atmospheric_transmissivity,
        emissivity_gradient_array_s,
        emissivity_gradient_array_p,
        theta_vals,
        theta_weights,
        wavelength_array,
    ):
        """Method to compute the gradient of the power radiated by the atmosphere and absorbed by the cooling structure

        Attribute
        ---------
        None

        Returns
        -------
        _absorbed_solar_spectrum_gradient
        """

        _nth = len(theta_vals)
        _ngr = len(emissivity_gradient_array_p[0, 0, :])
        _absorbed_atmospheric_radiation_gradient = np.zeros(_ngr)

        # make sure we are getting the blackbody spectrum of the atmosphere
        # store the structure temperature
        _T_temp = self.temperature
        # update the structure temperature, which is the attribute that
        # is used for computing the blackbody spectrum in _compute_therml_spectrum
        self.temperature = self.atmospheric_temperature
        self._compute_therml_spectrum(
            wavelength_array, emissivity_gradient_array_p[0, :, 0]
        )
        # set the structure temperature back to _T_temp in case
        # one wants to compute the thermal emission of the structure again!
        self.temperature = _T_temp

        for i in range(0, _ngr):
            P_atm_prime = 0
            for j in range(0, _nth):
                # get the term that goes in the exponent of the atmospheric transmissivity
                _o_over_cos_t = 1 / np.cos(theta_vals[j])
                _emissivity_atm = (
                    np.ones(len(atmospheric_transmissivity))
                    - atmospheric_transmissivity**_o_over_cos_t
                )
                _TE_atm = (
                    self.blackbody_spectrum * _emissivity_atm * np.cos(theta_vals[j])
                )
                _absorbed_TE_spectrum = (
                    _TE_atm
                    * 0.5
                    * (
                        emissivity_gradient_array_p[j, :, i]
                        + emissivity_gradient_array_s[j, :, i]
                    )
                )
                _absorbed_TE = np.trapz(_absorbed_TE_spectrum, wavelength_array)
                P_atm_prime += (
                    2 * np.pi * _absorbed_TE * np.sin(theta_vals[j]) * theta_weights[j]
                )
            _absorbed_atmospheric_radiation_gradient[i] = P_atm_prime

        return _absorbed_atmospheric_radiation_gradient

    def _compute_solar_radiated_power(
        self, solar_spectrum, emissivity_array_s, emissivity_array_p, wavelength_array
    ):
        """Method to compute the solar power absorbed by the cooling structure

        Attributes
        ----------
        None

        Returns
        -------
        P_sun

        References
        ----------
        See Eq. (4) of https://www.nature.com/articles/nature13883
        """
        # compute the absorbed solar spectrum
        _absorbed_solar_spectrum = (
            solar_spectrum * 0.5 * (emissivity_array_p + emissivity_array_s)
        )
        # integrate it!
        P_sun = np.trapz(_absorbed_solar_spectrum, wavelength_array)
        return P_sun

    def _compute_solar_radiated_power_gradient(
        self,
        solar_spectrum,
        emissivity_gradient_array_s,
        emissivity_gradient_array_p,
        wavelength_array,
    ):
        """Method to compute the gradient of the solar power absorbed by the cooling structure

        Attributes
        ----------
        None

        Returns
        -------
        _absorbed_solar_spectrum_gradient

        References
        ----------
        See Eq. (4) of https://www.nature.com/articles/nature13883
        """
        # get the dimension of the gradient vector
        _ngr = len(emissivity_gradient_array_s[0, :])
        _absorbed_solar_spectrum_gradient = np.zeros(_ngr)
        for i in range(0, _ngr):
            _absorbed_solar_spectrum_prime = (
                solar_spectrum
                * 0.5
                * (
                    emissivity_gradient_array_p[:, i]
                    + emissivity_gradient_array_s[:, i]
                )
            )
            # integrate it!
            _absorbed_solar_spectrum_gradient[i] = np.trapz(
                _absorbed_solar_spectrum_prime, wavelength_array
            )
        return _absorbed_solar_spectrum_gradient
