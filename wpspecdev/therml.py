import numpy as np
from scipy.interpolate import UnivariateSpline


class Therml:
    """Compute the absorption, scattering, and extinction spectra of a sphere using Mie theory

    Attributes
    ----------

    Returns
    -------
        None

    """

    def __init__(self, args):
        """constructor for the Therml class"""
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
            self.lambda_bandgap = 2250e-9

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
        # speed of light in SI
        c = 299792458
        # plancks constant in SI
        h = 6.62607004e-34
        # boltzmanns constant in SI
        kb = 1.38064852e-23

        self.blackbody_spectrum = 2 * h * c ** 2 / wavelength_array ** 5
        self.blackbody_spectrum /= (
            np.exp(h * c / (wavelength_array * kb * self.temperature)) - 1
        )
        self.thermal_emission_array = self.blackbody_spectrum * emissivity_array

    def _compute_power_density(self, wavelength_array):
        """method to compute the power density from blackbody spectrum and thermal emission spectrum

        Attributes
        ----------
            blackbody_power_density : float
                total power density radiated into a hemisphere of a blackbody at a given temperature

            power_density : float
                total power density radiated by structure at a given temperature

        References
        ----------
            Equation (15) and (16) of https://github.com/FoleyLab/wptherml/blob/master/docs/Equations.pdf

        """
        # fit cubic spline to blackbody spectrum
        blackbody_spline = UnivariateSpline(
            wavelength_array, self.blackbody_spectrum, k=3
        )

        # fit cubic spline to thermal emission spectrum
        thermal_emission_spline = UnivariateSpline(
            wavelength_array, self.thermal_emission_array
        )

        # get upper- and lower-bounds of integration
        a = wavelength_array[0]
        b = wavelength_array[len(wavelength_array) - 1]

        # integrate the spectra over wavelength
        self.blackbody_power_density = blackbody_spline.integral(a, b)

        self.power_density = thermal_emission_spline.integral(a, b)

        # account for angular integrals over hemisphere (assuming no angle dependence of emissivity)
        self.blackbody_power_density *= np.pi
        self.power_density *= np.pi

        # compute Blackbody power density from Stefan-Boltzmann law for validation
        # Stefan-Boltzmann constant
        sig = 5.670374419e-8
        self.stefan_boltzmann_law = sig * self.temperature ** 4

    def _compute_photopic_luminosity(wavelength_array):
        """computes the photopic luminosity function from a Gaussian fit

        Arguments
        ---------
            wavelength_array : numpy array of floats
                the array of wavelengths over which you will compute the photopic luminosity

        Attributes
        ----------
            photopic_luminosity_array : numpy array of floats
                the array of photopic luminosity values corresponding to each value of wavelength_array

        References
        ----------
            Data taken from http://www.cvrl.org/database/data/lum/linCIE2008v2e_5.htm
        """
        a = 1.02433
        b = 2.59462e14
        c = 5.60186e-07

        self.photopic_luminosity_array = a * np.exp(-b * (wavelength_array - c) ** 2)

    def _compute_stpv_power_density(wavelength_array):
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
        ### JJF Comments: I suggest breaking this calculation into a few steps:
        
        # 1. compute an intermediate array that is the product of self.thermal_emission_array and wavelength_array / self.lambda_bandgap
        # 2. fit a spline to this array similar to line 106 in _compute_power_density()
        # 3. find the lower limit of the integral from wavelength_array[0]; the upper limit of the integral is self.lambda_bandgap
        # 4. use the built-in integral method of your spline to integrate from the lower limit to the upper limit, storing the result to self.stpv_power_density
        # 5. multiply self.stpv_power_density by np.pi (note np.pi is the built-in variable for pi; I don't think np.pi() returns the value of pi.
        
        return np.pi()/self.lambda_bandgap*self.integrate(wavelength_array*\
            self.compute_therml_spectrum(wavelength_array, self.thermal_emission_array))

    def _compute_stpv_spectral_efficiency(wavelength_array):
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
        pass

    def _compute_luminous_efficiency(wavelength_array):
        """method to compute the luminous efficiency for an incandescent from the thermal emission spectrum of a structure

        Arguments
        ---------
            wavelength_array : numpy array of floats
                the wavelengths over which the thermal emission spectrum is known

        Attributes
        ----------

            thermal_emission_array : numpy array of floats (already assigned)
                the thermal emission spectrum for each value of wavelength array for the stpv structure

            photopic_luminosity_array : numpy array of floats (can be computed by calling self._compute_photopic_luminosity(wavelength_array))
                photopic luminosity function values corresponding to wavelength_array

            spectral_efficiency : float (will be computed by this function)
                the spectral efficiency of the incandescent source

        References
        ----------
            Equation (27) of https://github.com/FoleyLab/wptherml/blob/master/docs/Equations.pdf

        """
        pass
