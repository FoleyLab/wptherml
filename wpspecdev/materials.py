import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import os

path_and_file = os.path.realpath(__file__)
path = path_and_file[:-12]


class Materials:
    """Compute the absorption, scattering, and extinction spectra of a sphere using Mie theory"""

    def _create_test_multilayer(self, central_wavelength):
        """
        Simple method to create a 3-entry array of wavelengths as follows:
        [central_wavelength-1e-9 m, central_wavelength m, central_wavelength+1e-9 m]
        and dummy _refractive_index_array that can be filled in
        with actual materials at the wavelength arrays.
        This is simply meant to enable unit testing for desired wavelengths of the
        various materials methods
        """
        self.wavelength_array = np.array(
            [central_wavelength - 1e-9, central_wavelength, central_wavelength + 1e-9]
        )
        self.number_of_wavelengths = 3
        self.number_of_layers = 3
        self._refractive_index_array = np.reshape(
            np.tile(np.array([1 + 0j, 1 + 0j, 1 + 0j]), self.number_of_wavelengths),
            (self.number_of_wavelengths, self.number_of_layers),
        )

    def material_Air(self, layer_number):
        self._refractive_index_array[:, layer_number] = (
            np.ones(len(self.wavelength_array), dtype=complex) * 1.0
        )

    def material_SiO2(self, layer_number):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            """defines the refractive index of layer layer_number to be SiO2

            Arguments
            ----------
            layer_number : int
            specifies the layer of the stack that will be modelled as SiO2

            wavelength_range (optional) : str
            specifies wavelength regime that is desired for modelling the material

            Attributes
            ----------
            _refractive_index_array : 1 x number_of_wavelengths numpy array of complex floats

            Returns
            -------
            None

            Examples
            --------
            >>> material_SiO2(1, wavelength_range="visible") -> layer 1 will be SiO2 from the SiO2_ir
            data set good from visible to 50 microns (0.21-50)
            """
            # get path to the sio2 data file
            file_path = path + "data/SiO2_ir.txt"
            # now read SiO2 data into a numpy array
            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index
            n_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 1], k=1
            )
            k_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 2], k=1
            )

            self._refractive_index_array[:, layer_number] = n_spline(
                self.wavelength_array
            ) + 1j * k_spline(self.wavelength_array)

    def material_TiO2(self, layer_number):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            """defines the refractive index of layer layer_number to be TiO2

            Arguments
            ----------
            layer_number : int
            specifies the layer of the stack that will be modelled as TiO2

            wavelength_range (optional) : str
            specifies wavelength regime that is desired for modelling the material

            Attributes
            ----------
            _refractive_index_array : 1 x number_of_wavelengths numpy array of complex floats

            Returns
            -------
            None

            Examples
            --------
            >>> material_TiO2(1, wavelength_range="visible") -> layer 1 will be TiO2 from the Siefke data set good from visible to 125.123 microns (0.12-125.123)
            """
            # get path to the TiO2 data file
            file_path = path + "data/TiO2_Siefke.txt"
            # now read TiO2 data into a numpy array
            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index
            n_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 1], k=1
            )
            k_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 2], k=1
            )

            self._refractive_index_array[:, layer_number] = n_spline(
                self.wavelength_array
            ) + 1j * k_spline(self.wavelength_array)

    def material_Ta2O5(self, layer_number, wavelength_range="visible", override="true"):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            """defines the refractive index of layer layer_number to be Ta2O5

            Arguments
            ----------
            layer_number : int
            specifies the layer of the stack that will be modelled as Ta2O5

            wavelength_range (optional) : str
            specifies wavelength regime that is desired for modelling the material

            Attributes
            ----------
            _refractive_index_array : 1 x number_of_wavelengths numpy array of complex floats

            Returns
            -------
            None

            Examples
            --------
            >>> material_Ta2O5(1, wavelength_range="visible") -> layer 1 will be Ta2O5 from the Rodriguez data set good from visible to 1.5 microns (0.02-1.5)
            >>> material_Ta2O5(2, wavelength_range="ir") -> layer 2 will be Ta2O5 from the Bright data set good until 1000 microns (0.5-1000)
            """

            # dictionary specific to Ta2O5 with wavelength range information corresponding to different
            # data sets
            data1 = {
                "file": "data/Ta2O5_Rodriguez.txt",
                "lower_wavelength": 2.9494e-08,
                "upper_wavelength": 1.5143e-06,
            }
            data2 = {
                "file": "data/Ta2O5_Bright.txt",
                "lower_wavelength": 5.0000e-07,
                "upper_wavelength": 1.0000e-03,
            }

            shortest_wavelength = self.wavelength_array[0]
            longest_wavelength = self.wavelength_array[self.number_of_wavelengths - 1]

            if (
                shortest_wavelength >= data1["lower_wavelength"]
                and longest_wavelength <= data1["upper_wavelength"]
            ):
                file_path = path + data1["file"]

            elif (
                shortest_wavelength >= data2["lower_wavelength"]
                and longest_wavelength <= data2["upper_wavelength"]
            ):
                file_path = path + data2["file"]

            else:
                file_path = path + data1["file"]

            if override == "false":
                # make sure the wavelength_range string is all  lowercase
                wavelength_range = wavelength_range.lower()
                if (
                    wavelength_range == "visible"
                    or wavelength_range == "short"
                    or wavelength_range == "vis"
                ):
                    file_path = path + "data/Ta2O5_Rodriguez.txt"

                elif wavelength_range == "ir" or wavelength_range == "long":
                    file_path = path + "data/Ta2O5_Bright.txt"

            print("read from ", file_path)
            # now read Ta2O5 data into a numpy array
            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index
            n_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 1], k=1
            )
            k_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 2], k=1
            )

            self._refractive_index_array[:, layer_number] = n_spline(
                self.wavelength_array
            ) + 1j * k_spline(self.wavelength_array)

    def material_TiN(self, layer_number):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            """defines the refractive index of layer layer_number to be Tin

            Arguments
            ----------
            layer_number : int
            specifies the layer of the stack that will be modelled as Tin

            wavelength_range (optional) : str
            specifies wavelength regime that is desired for modelling the material

            Attributes
            ----------
            _refractive_index_array : 1 x number_of_wavelengths numpy array of complex floats

            Returns
            -------
            None

            Examples
            --------
            >>> material_Tin(1, wavelength_range="visible") -> layer 1 will be Tin from the Tin_ellipsometry data set good from visible to 7 microns (0.4-7)
            """
            # get path to the tin data file
            file_path = path + "data/TiN_ellipsometry_data.txt"
            # now read Tin data into a numpy array
            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index
            n_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 1], k=1
            )
            k_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 2], k=1
            )

            self._refractive_index_array[:, layer_number] = n_spline(
                self.wavelength_array
            ) + 1j * k_spline(self.wavelength_array)

    def material_static_refractive_index(self, layer_number, refractive_index):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            self._refractive_index_array[:, layer_number] = (
                np.ones(len(self.wavelength_array), dtype=complex) * refractive_index
            )

    def material_Al(self, layer_number):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            """defines the refractive index of layer layer_number to be Al

            Arguments
            ----------
            layer_number : int
            specifies the layer of the stack that will be modelled as Al

            wavelength_range (optional) : str
            specifies wavelength regime that is desired for modelling the material

            Attributes
            ----------
            _refractive_index_array : 1 x number_of_wavelengths numpy array of complex floats

            Returns
            -------
            None

            Examples
            --------
            >>> material_Al(1, wavelength_range="visible") -> layer 1 will be Al from the Al_Rakic data set good from visible to 200 microns (0.00012399-200)
            """
            # get path to the Al data file
            file_path = path + "data/Al_Rakic.txt"
            # now read Al data into a numpy array
            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index
            n_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 1], k=1
            )
            k_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 2], k=1
            )

            self._refractive_index_array[:, layer_number] = n_spline(
                self.wavelength_array
            ) + 1j * k_spline(self.wavelength_array)

    def material_W(self, layer_number):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            """defines the refractive index of layer layer_number to be W

            Arguments
            ----------
            layer_number : int
            specifies the layer of the stack that will be modelled as W

            wavelength_range (optional) : str
            specifies wavelength regime that is desired for modelling the material

            Attributes
            ----------
            _refractive_index_array : 1 x number_of_wavelengths numpy array of complex floats

            Returns
            -------
            None

            Examples
            --------
            >>> material_W(1, wavelength_range="visible") -> layer 1 will be W from the W_Ordal data set good from visible to 200 microns (0.667-200)
            """

            # get path to the W data file
            file_path = path + "data/W_Ordal.txt"
            # now read W data into a numpy array
            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index
            n_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 1], k=1
            )
            k_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 2], k=1
            )
            self._refractive_index_array[:, layer_number] = n_spline(
                self.wavelength_array
            ) + 1j * k_spline(self.wavelength_array)

    def material_Pt(self, layer_number):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            """defines the refractive index of layer layer_number to be Pt

            Arguments
            ----------
            layer_number : int
            specifies the layer of the stack that will be modelled as Pt

            wavelength_range (optional) : str
            specifies wavelength regime that is desired for modelling the material

            Attributes
            ----------
            _refractive_index_array : 1 x number_of_wavelengths numpy array of complex floats

            Returns
            -------
            None

            Examples
            --------
            >>> material_Pt(1, wavelength_range="visible") -> layer 1 will be Pt from the Pt_Rakic data set good from visible to 12.398 microns (0.24797-12.398)
            """
            self._refractive_index_array[:, layer_number] = (
                np.ones(len(self.wavelength_array), dtype=complex) * 2.4
            )
            # get path to the Platinum data file
            file_path = path + "data/Pt_Rakic.txt"
            # now read Platinum data into a numpy array
            file_data = np.loadtxt(file_path)
            n_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 1], k=1
            )
            k_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 2], k=1
            )
            self._refractive_index_array[:, layer_number] = n_spline(
                self.wavelength_array
            ) + 1j * k_spline(self.wavelength_array)

    def material_AlN(self, layer_number):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            """defines the refractive index of layer layer_number to be AlN

            Arguments
            ----------
            layer_number : int
            specifies the layer of the stack that will be modelled as AlN

            wavelength_range (optional) : str
            specifies wavelength regime that is desired for modelling the material

            Attributes
            ----------
            _refractive_index_array : 1 x number_of_wavelengths numpy array of complex floats

            Returns
            -------
            None

            Examples
            --------
            >>> material_AlN(1, wavelength_range="visible") -> layer 1 will be AlN from the AlN_Kischkat data set good from 1.53846 to 14.2857 microns
            """
            # get path to the AlN data file
            file_path = path + "data/AlN_Kischkat.txt"
            # now read AlN data into a numpy array

            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index
            n_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 1], k=1
            )
            k_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 2], k=1
            )
            self._refractive_index_array[:, layer_number] = n_spline(
                self.wavelength_array
            ) + 1j * k_spline(self.wavelength_array)

    def material_Pb(self, layer_number):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            """defines the refractive index of layer layer_number to be Pb

            Arguments
            ----------
            layer_number : int
            specifies the layer of the stack that will be modelled as Pb

            wavelength_range (optional) : str
            specifies wavelength regime that is desired for modelling the material

            Attributes
            ----------
            _refractive_index_array : 1 x number_of_wavelengths numpy array of complex floats

            Returns
            -------
            None

            Examples
            --------
            >>> material_Pb(1, wavelength_range="visible") -> layer 1 will be Pb from the Pb_Werner data set good from visible to 2.48 microns (0.0176-2.48)
            """
            self._refractive_index_array[:, layer_number] = (
                np.ones(len(self.wavelength_array), dtype=complex) * 2.4
            )
            # get path to the Lead data file
            file_path = path + "data/Pb_Werner.txt"
            # now read Lead data into a numpy array
            file_data = np.loadtxt(file_path)
            n_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 1], k=1
            )
            k_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 2], k=1
            )
            self._refractive_index_array[:, layer_number] = n_spline(
                self.wavelength_array
            ) + 1j * k_spline(self.wavelength_array)

    def material_HfO2(self, layer_number):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            """defines the refractive index of layer layer_number to be HfO2

            Arguments
            ----------
            layer_number : int
            specifies the layer of the stack that will be modelled as HfO2

            wavelength_range (optional) : str
            specifies wavelength regime that is desired for modelling the material

            Attributes
            ----------
            _refractive_index_array : 1 x number_of_wavelengths numpy array of complex floats

            Returns
            -------
            None

            Examples
            --------
            >>> material_HfO2(1, wavelength_range="visible") -> layer 1 will be HfO2 from the HfO2_Al-Kuhaili data set good from visible to 2 microns (0.2-2)
            """
            self._refractive_index_array[:, layer_number] = (
                np.ones(len(self.wavelength_array), dtype=complex) * 2.4
            )
            # get path to the HfO2 data file
            file_path = path + "data/HfO2_Al-Kuhaili.txt"

            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index
            n_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 1], k=1
            )
            k_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 2], k=1
            )
            self._refractive_index_array[:, layer_number] = n_spline(
                self.wavelength_array
            ) + 1j * k_spline(self.wavelength_array)

    def material_Ag(self, layer_number):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            """defines the refractive index of layer layer_number to be Ag

            Arguments
            ----------
            layer_number : int
            specifies the layer of the stack that will be modelled as Ag

            wavelength_range (optional) : str
            specifies wavelength regime that is desired for modelling the material

            Attributes
            ----------
            _refractive_index_array : 1 x number_of_wavelengths numpy array of complex floats

            Returns
            -------
            None

            Examples
            --------
            >>> material_Ag(1, wavelength_range="visible") -> layer 1 will be Ag from the Ag_ri.txt data set good from visible to 200 microns (0.000123-200)
            """
            self._refractive_index_array[:, layer_number] = (
                np.ones(len(self.wavelength_array), dtype=complex) * 2.4
            )
            # get path to the Ag data file
            file_path = path + "data/Ag_ri.txt"
            # now read Ag data into a numpy array
            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index
            n_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 1], k=1
            )
            k_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 2], k=1
            )
            self._refractive_index_array[:, layer_number] = n_spline(
                self.wavelength_array
            ) + 1j * k_spline(self.wavelength_array)

    def material_Re(self, layer_number):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            """defines the refractive index of layer layer_number to be Re

            Arguments
            ----------
            layer_number : int
            specifies the layer of the stack that will be modelled as Re

            wavelength_range (optional) : str
            specifies wavelength regime that is desired for modelling the material

            Attributes
            ----------
            _refractive_index_array : 1 x number_of_wavelengths numpy array of complex floats

            Returns
            -------
            None

            Examples
            --------
            >>> material_Re(1, wavelength_range="visible") -> layer 1 will be Re from the Re_Windt data set good from 0.0024 to 0.1220 microns
            """
            self._refractive_index_array[:, layer_number] = (
                np.ones(len(self.wavelength_array), dtype=complex) * 2.4
            )
            # get path to the Re data file
            file_path = path + "data/Re_Windt.txt"
            # now read Lead data into a numpy array
            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index
            n_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 1], k=1
            )
            k_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 2], k=1
            )
            self._refractive_index_array[:, layer_number] = n_spline(
                self.wavelength_array
            ) + 1j * k_spline(self.wavelength_array)

    def material_Au(self, layer_number, wavelength_range="visible", override="true"):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            """defines the refractive index of layer layer_number to be Au

            Arguments
            ----------
            layer_number : int
            specifies the layer of the stack that will be modelled as Au
            wavelength_range (optional) : str
            specifies wavelength regime that is desired for modelling the material

            Attributes
            ----------
            _refractive_index_array : 1 x number_of_wavelengths numpy array of complex floats

            Returns
            -------
            None

            Examples
            --------
            >>> material_Au(1, wavelength_range="visible") -> layer 1 will be Au from the JC_RI_f data set  from 0.2 to 1.000025 microns
            >>> material_Au(2, wavelength_range="ir") -> layer 2 will be Au from the Au_IR data set from 0.3 to 24.93 microns
            """

            # dictionary specific to Au with wavelength range information corresponding to different
            # data sets
            data1 = {
                "file": "data/Au_JC_RI_f.txt",
                "lower_wavelength": 2e-07,
                "upper_wavelength": 1.00025e-06,
            }
            data2 = {
                "file": "data/Au_IR.txt",
                "lower_wavelength": 3.000000e-07,
                "upper_wavelength": 2.493000e-05,
            }

            shortest_wavelength = self.wavelength_array[0]
            longest_wavelength = self.wavelength_array[self.number_of_wavelengths - 1]

            if (
                shortest_wavelength >= data1["lower_wavelength"]
                and longest_wavelength <= data1["upper_wavelength"]
            ):
                file_path = path + data1["file"]

            elif (
                shortest_wavelength >= data2["lower_wavelength"]
                and longest_wavelength <= data2["upper_wavelength"]
            ):
                file_path = path + data2["file"]

            else:
                file_path = path + data1["file"]

            if override == "false":
                # make sure the wavelength_range string is all  lowercase
                wavelength_range = wavelength_range.lower()
                if (
                    wavelength_range == "visible"
                    or wavelength_range == "short"
                    or wavelength_range == "vis"
                ):
                    file_path = path + "data/Au_JC_RI_f.txt.txt"

                elif wavelength_range == "ir" or wavelength_range == "long":
                    file_path = path + "data/Au_IR.txt"

            print("read from ", file_path)
            # now read Au data into a numpy array
            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index
            n_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 1], k=1
            )
            k_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 2], k=1
            )

            self._refractive_index_array[:, layer_number] = n_spline(
                self.wavelength_array
            ) + 1j * k_spline(self.wavelength_array)

    def material_Rh(self, layer_number):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            """defines the refractive index of layer layer_number to be Rh

            Arguments
            ----------
            layer_number : int
            specifies the layer of the stack that will be modelled as Rh

            wavelength_range (optional) : str
            specifies wavelength regime that is desired for modelling the material

            Attributes
            ----------
            _refractive_index_array : 1 x number_of_wavelengths numpy array of complex floats

            Returns
            -------
            None

            Examples
            --------
            >>> material_Rh(1, wavelength_range="visible") -> layer 1 will be Rh from the Rh_Weaver data set good from visible to 12.4 microns (0.2-12.4)
            """
            self._refractive_index_array[:, layer_number] = (
                np.ones(len(self.wavelength_array), dtype=complex) * 2.4
            )
            # get path to the Rh data file
            file_path = path + "data/Rh_Weaver.txt"
            # now read Rh data into a numpy array
            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index
            n_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 1], k=1
            )
            k_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 2], k=1
            )
            self._refractive_index_array[:, layer_number] = n_spline(
                self.wavelength_array
            ) + 1j * k_spline(self.wavelength_array)

    def material_Al2O3(self, layer_number):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            """defines the refractive index of layer layer_number to be Al2O3

            Arguments
            ----------
            layer_number : int
            specifies the layer of the stack that will be modelled as Al2O3

            wavelength_range (optional) : str
            specifies wavelength regime that is desired for modelling the material

            Attributes
            ----------
            _refractive_index_array : 1 x number_of_wavelengths numpy array of complex floats

            Returns
            -------
            None

            Examples
            --------
            >>> material_Al2O3(1, wavelength_range="visible") -> layer 1 will be Al2O3 from the Al2O3_ri.txt data set good from visible to 2.5 microns (0.4-2.5)
            """
            self._refractive_index_array[:, layer_number] = (
                np.ones(len(self.wavelength_array), dtype=complex) * 2.4
            )
            # get path to the Al2O3 data file
            file_path = path + "data/Al2O3_ri.txt"
            # now read Au data into a numpy array
            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index
            n_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 1], k=1
            )
            k_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 2], k=1
            )
            self._refractive_index_array[:, layer_number] = n_spline(
                self.wavelength_array
            ) + 1j * k_spline(self.wavelength_array)

    def material_Ru(self, layer_number):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            """defines the refractive index of layer layer_number to be Ru

            Arguments
            ----------
            layer_number : int
            specifies the layer of the stack that will be modelled as Ru

            wavelength_range (optional) : str
            specifies wavelength regime that is desired for modelling the material

            Attributes
            ----------
            _refractive_index_array : 1 x number_of_wavelengths numpy array of complex floats

            Returns
            -------
            None

            Examples
            --------
            >>> material_Ru(1, wavelength_range="visible") -> layer 1 will be Ru from the Ru.txt data set good from visible to 6.0 microns (0.4-6.0)
            """
            self._refractive_index_array[:, layer_number] = (
                np.ones(len(self.wavelength_array), dtype=complex) * 2.4
            )
            # get path to the Ru data file
            file_path = path + "data/Ru.txt"
            # now read Ru data into a numpy array
            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index
            n_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 1], k=1
            )
            k_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 2], k=1
            )

            self._refractive_index_array[:, layer_number] = n_spline(
                self.wavelength_array
            ) + 1j * k_spline(self.wavelength_array)

    def material_polystyrene(self, layer_number):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            """defines the refractive index of layer layer_number to be polystyrene

            Arguments
            ----------
            layer_number : int
            specifies the layer of the stack that will be modelled as polystyrene

            wavelength_range (optional) : str
            specifies wavelength regime that is desired for modelling the material

            Attributes
            ----------
            _refractive_index_array : 1 x number_of_wavelengths numpy array of complex floats

            Returns
            -------
            None

            Examples
            --------
            >>> material_polystyrene(1, wavelength_range="visible") -> layer 1 will be polystyrene from the polystyrene.txt data set good from visible to 19.942 microns (0.4-19.942)
            """
            self._refractive_index_array[:, layer_number] = (
                np.ones(len(self.wavelength_array), dtype=complex) * 2.4
            )
            # get path to the polystyrene data file
            file_path = path + "data/Polystyrene.txt"
            # now read Au data into a numpy array
            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index
            n_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 1], k=1
            )
            k_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 2], k=1
            )
            self._refractive_index_array[:, layer_number] = n_spline(
                self.wavelength_array
            ) + 1j * k_spline(self.wavelength_array)

    def material_AlN(self, layer_number, wavelength_range="visible", override="true"):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            """defines the refractive index of layer layer_number to be AlN

            Arguments
            ----------
            layer_number : int
            specifies the layer of the stack that will be modelled as AlN

            wavelength_range (optional) : str
            specifies wavelength regime that is desired for modelling the material

            Attributes
            ----------
            _refractive_index_array : 1 x number_of_wavelengths numpy array of complex floats

            Returns
            -------
            None

            Examples
            --------
            >>> material_AlN(1, wavelength_range="visible") -> layer 1 will be AlN from the AlN_Pastrnak data set from 0.22 to 5.0 microns
            >>> material_AlN(2, wavelength_range="ir") -> layer 2 will be AlN from the AlN_Kischkat data set from 1.53846 to 14.2857 microns
            """

            # dictionary specific to AlN with wavelength range information corresponding to different
            # data sets
            data1 = {
                "file": "data/AlN_Pastrnak.txt",
                "lower_wavelength": 0.22e-6,
                "upper_wavelength": 5.00e-6,
            }
            data2 = {
                "file": "data/AlN_Kischkat.txt",
                "lower_wavelength": 1.53846e-06,
                "upper_wavelength": 14.2857e-06,
            }

            shortest_wavelength = self.wavelength_array[0]
            longest_wavelength = self.wavelength_array[self.number_of_wavelengths - 1]

            if (
                shortest_wavelength >= data1["lower_wavelength"]
                and longest_wavelength <= data1["upper_wavelength"]
            ):
                file_path = path + data1["file"]
                print("1")
            elif (
                shortest_wavelength >= data2["lower_wavelength"]
                and longest_wavelength <= data2["upper_wavelength"]
            ):
                file_path = path + data2["file"]
                print("2")
            else:
                file_path = path + data1["file"]

            if override == "false":
                # make sure the wavelength_range string is all  lowercase
                wavelength_range = wavelength_range.lower()
                if (
                    wavelength_range == "visible"
                    or wavelength_range == "short"
                    or wavelength_range == "vis"
                ):
                    file_path = path + "data/AlN_Pastrnak.txt"

                elif wavelength_range == "ir" or wavelength_range == "long":
                    file_path = path + "data/AlN_Kischkat.txt"

            print("read from ", file_path)
            # now read AlN data into a numpy array
            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index
            n_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 1], k=1
            )
            k_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 2], k=1
            )

            self._refractive_index_array[:, layer_number] = n_spline(
                self.wavelength_array
            ) + 1j * k_spline(self.wavelength_array)

    def material_W(self, layer_number, wavelength_range="visible", override="true"):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            """defines the refractive index of layer layer_number to be W

            Arguments
            ----------
            layer_number : int
            specifies the layer of the stack that will be modelled as W

            wavelength_range (optional) : str
            specifies wavelength regime that is desired for modelling the material

            Attributes
            ----------
            _refractive_index_array : 1 x number_of_wavelengths numpy array of complex floats

            Returns
            -------
            None

            Examples
            --------
            >>> material_W(1, wavelength_range="visible") -> layer 1 will be W from the W_Rakic data set good from visible to 12.398 microns (0.24797-12.398)
            >>> material_W(2, wavelength_range="ir") -> layer 2 will be W from the Ordal data set good until 200 microns (0.667-200)
            """

            # dictionary specific to W with wavelength range information corresponding to different
            # data sets
            data1 = {
                "file": "data/W_Rakic.txt",
                "lower_wavelength": 2.4797e-07,
                "upper_wavelength": 1.2398e-05,
            }
            data2 = {
                "file": "data/W_Ordal.txt",
                "lower_wavelength": 6.67000e-07,
                "upper_wavelength": 2.00000e-04,
            }

            shortest_wavelength = self.wavelength_array[0]
            longest_wavelength = self.wavelength_array[self.number_of_wavelengths - 1]

            if (
                shortest_wavelength >= data1["lower_wavelength"]
                and longest_wavelength <= data1["upper_wavelength"]
            ):
                file_path = path + data1["file"]

            elif (
                shortest_wavelength >= data2["lower_wavelength"]
                and longest_wavelength <= data2["upper_wavelength"]
            ):
                file_path = path + data2["file"]

            else:
                file_path = path + data1["file"]

            if override == "false":
                # make sure the wavelength_range string is all  lowercase
                wavelength_range = wavelength_range.lower()
                if (
                    wavelength_range == "visible"
                    or wavelength_range == "short"
                    or wavelength_range == "vis"
                ):
                    file_path = path + "data/W_Rakic.txt"

                elif wavelength_range == "ir" or wavelength_range == "long":
                    file_path = path + "data/W_Ordal.txt"

            print("read from ", file_path)
            # now read W data into a numpy array
            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index
            n_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 1], k=1
            )
            k_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 2], k=1
            )

            self._refractive_index_array[:, layer_number] = n_spline(
                self.wavelength_array
            ) + 1j * k_spline(self.wavelength_array)

    def material_Si(self, layer_number, wavelength_range="visible", override="true"):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            """defines the refractive index of layer layer_number to be Si

            Arguments
            ----------
            layer_number : int
            specifies the layer of the stack that will be modelled as Si

            wavelength_range (optional) : str
            specifies wavelength regime that is desired for modelling the material

            Attributes
            ----------
            _refractive_index_array : 1 x number_of_wavelengths numpy array of complex floats

            Returns
            -------
            None

            Examples
            --------
            >>> material_si(1, wavelength_range="visible") -> layer 1 will be Si from the Schinke data set good from visible to 1.5 microns (0.25-1.45)
            >>> material_si(2, wavelength_range="ir") -> layer 2 will be Si from the Shkondin data set good until 1000 microns (2-20)
            """

            # dictionary specific to W with wavelength range information corresponding to different
            # data sets
            data1 = {
                "file": "data/Si_Schinke.txt",
                "lower_wavelength": 0.000000250,
                "upper_wavelength": 0.000001450,
            }
            data2 = {
                "file": "data/Si_Shkondin.txt",
                "lower_wavelength": 0.00000200000,
                "upper_wavelength": 0.00002000000,
            }

            shortest_wavelength = self.wavelength_array[0]
            longest_wavelength = self.wavelength_array[self.number_of_wavelengths - 1]

            if (
                shortest_wavelength >= data1["lower_wavelength"]
                and longest_wavelength <= data1["upper_wavelength"]
            ):
                file_path = path + data1["file"]

            elif (
                shortest_wavelength >= data2["lower_wavelength"]
                and longest_wavelength <= data2["upper_wavelength"]
            ):
                file_path = path + data2["file"]

            else:
                file_path = path + data1["file"]

            if override == "false":
                # make sure the wavelength_range string is all  lowercase
                wavelength_range = wavelength_range.lower()
                if (
                    wavelength_range == "visible"
                    or wavelength_range == "short"
                    or wavelength_range == "vis"
                ):
                    file_path = path + "data/Si_Schinke.txt"

                elif wavelength_range == "ir" or wavelength_range == "long":
                    file_path = path + "data/Si_Shkondin.txt"

            print("read from ", file_path)
            # now read Si data into a numpy array
            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index
            n_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 1], k=1
            )
            k_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 2], k=1
            )

            self._refractive_index_array[:, layer_number] = n_spline(
                self.wavelength_array
            ) + 1j * k_spline(self.wavelength_array)

    def material_Re(self, layer_number, wavelength_range="visible", override="true"):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            """defines the refractive index of layer layer_number to be Re

            Arguments
            ----------
            layer_number : int
            specifies the layer of the stack that will be modelled as Re

            wavelength_range (optional) : str
            specifies wavelength regime that is desired for modelling the material

            Attributes
            ----------
            _refractive_index_array : 1 x number_of_wavelengths numpy array of complex floats

            Returns
            -------
            None

            Examples
            --------
            >>> material_Re(1, wavelength_range="visible") -> layer 1 will be Re from the Re_Windt data set good from visible to 1.5 microns (0.00236-0.12157)
            >>> material_Re(2, wavelength_range="ir") -> layer 2 will be Re from the Re_Palik data set good until 6 microns (0.4-6)
            """

            # dictionary specific to W with wavelength range information corresponding to different
            # data sets
            data1 = {
                "file": "data/Re_Windt.txt",
                "lower_wavelength": 2.36e-09,
                "upper_wavelength": 1.2157e-07,
            }
            data2 = {
                "file": "data/Re_Palik.txt",
                "lower_wavelength": 0.0000004000,
                "upper_wavelength": 0.0000060000,
            }

            shortest_wavelength = self.wavelength_array[0]
            longest_wavelength = self.wavelength_array[self.number_of_wavelengths - 1]

            if (
                shortest_wavelength >= data1["lower_wavelength"]
                and longest_wavelength <= data1["upper_wavelength"]
            ):
                file_path = path + data1["file"]

            elif (
                shortest_wavelength >= data2["lower_wavelength"]
                and longest_wavelength <= data2["upper_wavelength"]
            ):
                file_path = path + data2["file"]

            else:
                file_path = path + data1["file"]

            if override == "false":
                # make sure the wavelength_range string is all  lowercase
                wavelength_range = wavelength_range.lower()
                if (
                    wavelength_range == "visible"
                    or wavelength_range == "short"
                    or wavelength_range == "vis"
                ):
                    file_path = path + "data/Re_Windt.txt"

                elif wavelength_range == "ir" or wavelength_range == "long":
                    file_path = path + "data/Re_Palik.txt"

            print("read from ", file_path)
            # now read Re data into a numpy array
            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index
            n_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 1], k=1
            )
            k_spline = InterpolatedUnivariateSpline(
                file_data[:, 0], file_data[:, 2], k=1
            )

            self._refractive_index_array[:, layer_number] = n_spline(
                self.wavelength_array
            ) + 1j * k_spline(self.wavelength_array)


def material_Ag(self, layer_number, wavelength_range="visible", override="true"):
    if layer_number > 0 and layer_number < (self.number_of_layers - 1):
        """defines the refractive index of layer layer_number to be Ag

        Arguments
        ----------
        layer_number : int
        specifies the layer of the stack that will be modelled as Ag

        wavelength_range (optional) : str
        specifies wavelength regime that is desired for modelling the material

        Attributes
        ----------
        _refractive_index_array : 1 x number_of_wavelengths numpy array of complex floats

        Returns
        -------
        None

        Examples
        --------
        >>> material_Ag(1, wavelength_range="visible") -> layer 1 will be Ag from the Ag_JC data set good from 0.1879 to 1.937
        >>> material_Ag(2, wavelength_range="ir") -> layer 2 will be Ag from the Yang data set good until 24.92 microns (0.27-24.92)
        """

        # dictionary specific to W with wavelength range information corresponding to different
        # data sets
        data1 = {
            "file": "data/Ag_JC.txt",
            "lower_wavelength": 1.87900e-07,
            "upper_wavelength": 1.93700e-06,
        }
        data2 = {
            "file": "data/Ag_Yang.txt",
            "lower_wavelength": 2.70000e-07,
            "upper_wavelength": 2.49200e-05,
        }

        shortest_wavelength = self.wavelength_array[0]
        longest_wavelength = self.wavelength_array[self.number_of_wavelengths - 1]

        if (
            shortest_wavelength >= data1["lower_wavelength"]
            and longest_wavelength <= data1["upper_wavelength"]
        ):
            file_path = path + data1["file"]

        elif (
            shortest_wavelength >= data2["lower_wavelength"]
            and longest_wavelength <= data2["upper_wavelength"]
        ):
            file_path = path + data2["file"]

        else:
            file_path = path + data1["file"]

        if override == "false":
            # make sure the wavelength_range string is all  lowercase
            wavelength_range = wavelength_range.lower()
            if (
                wavelength_range == "visible"
                or wavelength_range == "short"
                or wavelength_range == "vis"
            ):
                file_path = path + "data/Ag_JC.txt"

            elif wavelength_range == "ir" or wavelength_range == "long":
                file_path = path + "data/Ag_Yang.txt"

        print("read from ", file_path)
        # now read Ag data into a numpy array
        file_data = np.loadtxt(file_path)
        # file_path[:,0] -> wavelengths in meters
        # file_path[:,1] -> real part of the refractive index
        # file_path[:,2] -> imaginary part of the refractive index
        n_spline = InterpolatedUnivariateSpline(file_data[:, 0], file_data[:, 1], k=1)
        k_spline = InterpolatedUnivariateSpline(file_data[:, 0], file_data[:, 2], k=1)

        self._refractive_index_array[:, layer_number] = n_spline(
            self.wavelength_array
        ) + 1j * k_spline(self.wavelength_array)
