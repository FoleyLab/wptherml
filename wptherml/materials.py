import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import os
from scipy import constants

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

    def _find_unique_ri_file_data(self, wl_array):
        """
        A simple method to eliminate redundant or numerically indistinguishable
        data from refractive index files.

        Arguments
        ---------
            wl_array : numpy array of floats
                the array of wavelengths read from a refractive index file

        Returns
        -------
            unique_index_array : array of ints
                the array of the indices of the unique elements of the
                refractive index file

        """

        wl_val = wl_array[0]
        unique_index_array = [0]

        for i in range(1, len(wl_array)):
            new_wl_val = wl_array[i]
            # skip over redundant values
            if new_wl_val <= wl_val:
                wl_val = new_wl_val
            else:
                unique_index_array.append(i)
                wl_val = new_wl_val

        return unique_index_array

    def material_H2O(self, layer_number):
        """defines the refractive index layer of layer_number to be water
        assuming static refractive index of n = 1.33 + 0j
        """
        self._refractive_index_array[:, layer_number] = (
            np.ones(len(self.wavelength_array), dtype=complex) * 1.33
        )

    def insert_layer(self, layer_number):
        """insert an air layer between layer_number-1 and layer_number
        e.g. if you have a structure that is Air/SiO2/HfO2/Ag/Air
        and you issue insert_layer(1), the new structure will be
        Air/Air/SiO2/HfO2/Ag/Air
        if you issue insert_layer(2), the new structure will be
        Air/SiO2/Air/HfO2/Ag/Air

        """
        _nwl = len(self._refractive_index_array[:, 0])
        _nl = len(self._refractive_index_array[0, :])
        _temp_ri_array = np.copy(self._refractive_index_array)
        _new_ri_array = np.zeros((_nwl, _nl + 1), dtype=complex)
        _new_air_layer = np.ones(_nwl, dtype=complex) * 1.0
        _new_ri_array[:, :layer_number] = _temp_ri_array[:, :layer_number]
        _new_ri_array[:, layer_number] = _new_air_layer
        _new_ri_array[:, layer_number + 1 :] = _temp_ri_array[:, layer_number:]
        self._refractive_index_array = np.copy(_new_ri_array)

    def material_Air(self, layer_number):
        """defines the refractive index layer of layer_number to be air
        assuming static refractive index of n = 1.0 + 0j
        """
        self._refractive_index_array[:, layer_number] = (
            np.ones(len(self.wavelength_array), dtype=complex) * 1.0
        )

    def material_from_file(self, layer_number, file_name):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            """defines the refractive index of layer layer_number to be
               from a text file with name "file_name" where the text file is ordered:
               column 1: wavelength in meters, increasing order
               column 2: real part of refractive index corresponding to wavelengths in col 1
               column 3: imaginary part of refractive index corresponding to wavelengths in col 1
               the file is expected to be in the directory $wpthermldir/wptherml/data
               where $wpthermldir is the full path to the directory where you have wptherml installed

            Arguments
            ----------
            layer_number : int
                specifies the layer of the stack that will be modelled as SiO2

            file_name : str
                the full file-name of the data file containing your refractive index information

            Attributes
            ----------
            _refractive_index_array : 1 x number_of_wavelengths numpy array of complex floats

            Returns
            -------
            None

            Examples
            --------
            >>> material_from_file(1, "SiO2_ir.txt") -> layer 1 will be SiO2 from the SiO2_ir
            data set good from visible to 50 microns (0.21-50)
            """
            # get path to the sio2 data file
            file_path = path + "data/" + file_name
            # now read SiO2 data into a numpy array
            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index
            # now read Ag data into a numpy array
            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index

            # sometimes there are duplicate wavelength, n, and k entries
            # in a data set; we want only the unique elements
            idx = self._find_unique_ri_file_data(file_data[:, 0])

            n_spline = InterpolatedUnivariateSpline(
                file_data[idx, 0], file_data[idx, 1], k=1
            )
            k_spline = InterpolatedUnivariateSpline(
                file_data[idx, 0], file_data[idx, 2], k=1
            )

            self._refractive_index_array[:, layer_number] = n_spline(
                self.wavelength_array
            ) + 1j * k_spline(self.wavelength_array)

    def material_2D_HOIP(self, layer_number):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            """defines the refractive index of layer layer_number to be 2D hybrid organic-inorganic
               perovskites

            Reference
            ---------
            Song et al. "Determination of dielectric functions and exciton oscillator
            strength of two-dimensional hybrid perovskites", ACS Materials Lett. 2021, 3, 1, 148-159

            Arguments
            ----------
            layer_number : int
            specifies the layer of the stack that will be modelled as SiO2

            Attributes
            ----------
            _refractive_index_array : 1 x number_of_wavelengths numpy array of complex floats

            Returns
            -------
            None
            """
            # get path to the sio2 data file
            file_path = path + "data/2D_HOIP.txt"
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

    def material_Si3N4(self, layer_number):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            """defines the refractive index of layer layer_number to be Si3N4

            Arguments
            ----------
            layer_number : int
            specifies the layer of the stack that will be modelled as Si3N4

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
            >>> material_Si3N4(1, wavelength_range="visible") -> layer 1 will be Si3N4 from the Si3N4_Luke.txt data set good from visible to 5 microns (0.361-5.14)
            """
            # get path to the Si3N4 data file
            file_path = path + "data/Si3N4_Luke.txt"
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

    def material_ZrO2(self, layer_number):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            """defines the refractive index of layer layer_number to be ZrO2

            Arguments
            ----------
            layer_number : int
            specifies the layer of the stack that will be modelled as ZrO2

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
            >>> material_ZrO2(1, wavelength_range="visible") -> layer 1 will be ZrO2 from the ZrO2_Wood.txt data set good from visible to 5.5 microns (0.31-5.504)
            """
            # get path to the Zr02 data file
            file_path = path + "data/ZrO2_Wood.txt"
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

    def material_SiO2_UDM(self, layer_number):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            """defines the refractive index of layer layer_number to be SiO2 using a universal dispersion model (UDM)

            Arguments
            ----------
            layer_number : int
            specifies the layer of the stack that will be modelled as SiO2 udm

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
            >>> material_SiO2_UDM(1, wavelength_range="visible") -> layer 1 will be SiO2 from the UDM data set good from 0.01 to 100 eV
            """
            # get path to the Si02 data file
            file_path = path + "data/SiO2_udm.txt"
            # now read Tin data into a numpy array
            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index
            e_data = file_data[:, 0]
            wl_nm = 1239.84193 / e_data

            wl_si = np.flip(wl_nm) * 1e-9

            n_array = np.flip(file_data[:,1])
            k_array = np.flip(file_data[:,2])

            n_spline = InterpolatedUnivariateSpline(
                wl_si, n_array, k=1
            )
            k_spline = InterpolatedUnivariateSpline(
                wl_si, k_array, k=1
            )

            self._refractive_index_array[:, layer_number] = n_spline(
                self.wavelength_array
            ) + 1j * k_spline(self.wavelength_array)

    def material_SiO2_UDM_v2(self, layer_number):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            """defines the refractive index of layer layer_number to be SiO2 using the universal dispersion model (UDM) here: http://newad.physics.muni.cz/table-udm/LithosilQ2-SPIE9890.Enk 

            Arguments
            ----------
            layer_number : int
            specifies the layer of the stack that will be modelled as SiO2 udm

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
            >>> material_SiO2_UDM_v2(1, wavelength_range="visible") -> layer 1 will be SiO2 from the UDM data set good from 0.01 to 100 eV
            """
            # get path to the Si02 data file
            file_path = path + "data/SiO2_udm_v2.txt"
            # now read Tin data into a numpy array
            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index
            e_data = file_data[:, 0]
            wl_nm = 1239.84193 / e_data

            wl_si = np.flip(wl_nm) * 1e-9

            n_array = np.flip(file_data[:,1])
            k_array = np.flip(file_data[:,2])

            n_spline = InterpolatedUnivariateSpline(
                wl_si, n_array, k=1
            )
            k_spline = InterpolatedUnivariateSpline(
                wl_si, k_array, k=1
            )

            self._refractive_index_array[:, layer_number] = n_spline(
                self.wavelength_array
            ) + 1j * k_spline(self.wavelength_array)

    def material_HfO2_UDM(self, layer_number):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            """defines the refractive index of layer layer_number to be HfO2 using the universal dispersion model (UDM) here: http://newad.physics.muni.cz/table-udm/HfO2-X2194-AO54_9108.Enk

            Arguments
            ----------
            layer_number : int
            specifies the layer of the stack that will be modelled as HfO2 udm

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
            >>> material_HfO2_UDM(1, wavelength_range="visible") -> layer 1 will be HfO2 from the UDM data set good from 0.01 to 100 eV
            """
            # get path to the Si02 data file
            file_path = path + "data/HfO2_udm.txt"
            # now read Tin data into a numpy array
            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index
            e_data = file_data[:, 0]
            wl_nm = 1239.84193 / e_data

            wl_si = np.flip(wl_nm) * 1e-9

            n_array = np.flip(file_data[:,1])
            k_array = np.flip(file_data[:,2])

            n_spline = InterpolatedUnivariateSpline(
                wl_si, n_array, k=1
            )
            k_spline = InterpolatedUnivariateSpline(
                wl_si, k_array, k=1
            )

            self._refractive_index_array[:, layer_number] = n_spline(
                self.wavelength_array
            ) + 1j * k_spline(self.wavelength_array)

    def material_HfO2_UDM_v2(self, layer_number):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            """defines the refractive index of layer layer_number to be HfO2 using the universal dispersion model (UDM) here: http://newad.physics.muni.cz/table-udm/HfO2-X2194-AO54_9108.Enk
               with the modification that the imaginary part of the refractive index is set to zero!

            Arguments
            ----------
            layer_number : int
            specifies the layer of the stack that will be modelled as HfO2 udm

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
            >>> material_HfO2_UDM(1, wavelength_range="visible") -> layer 1 will be "lossless" HfO2 from the UDM data set good from 0.01 to 100 eV
            """
            # get path to the Si02 data file
            file_path = path + "data/HfO2_udm_no_loss.txt"
            # now read Tin data into a numpy array
            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index
            e_data = file_data[:, 0]
            wl_nm = 1239.84193 / e_data

            wl_si = np.flip(wl_nm) * 1e-9

            n_array = np.flip(file_data[:,1])
            k_array = np.flip(file_data[:,2])

            n_spline = InterpolatedUnivariateSpline(
                wl_si, n_array, k=1
            )
            k_spline = InterpolatedUnivariateSpline(
                wl_si, k_array, k=1
            )

            self._refractive_index_array[:, layer_number] = n_spline(
                self.wavelength_array
            ) + 1j * k_spline(self.wavelength_array)

    def material_MgF2_UDM(self, layer_number):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            """defines the refractive index of layer layer_number to be MgF2 using the universal dispersion model (UDM) here: http://newad.physics.muni.cz/table-udm/MgF2-X2935-SPIE9628.Enk
               

            Arguments
            ----------
            layer_number : int
            specifies the layer of the stack that will be modelled as MgF2 udm

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
            >>> material_HfO2_UDM(1, wavelength_range="visible") -> layer 1 will be MgF2 from the UDM data set good from 0.01 to 100 eV
            """
            # get path to the Si02 data file
            file_path = path + "data/MgF2_udm.txt"
            # now read Tin data into a numpy array
            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index
            e_data = file_data[:, 0]
            wl_nm = 1239.84193 / e_data

            wl_si = np.flip(wl_nm) * 1e-9

            n_array = np.flip(file_data[:,1])
            k_array = np.flip(file_data[:,2])

            n_spline = InterpolatedUnivariateSpline(
                wl_si, n_array, k=1
            )
            k_spline = InterpolatedUnivariateSpline(
                wl_si, k_array, k=1
            )

            self._refractive_index_array[:, layer_number] = n_spline(
                self.wavelength_array
            ) + 1j * k_spline(self.wavelength_array)


    def material_Al2O3_UDM(self, layer_number):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            """defines the refractive index of layer layer_number to be Al203

            Arguments
            ----------
            layer_number : int
            specifies the layer of the stack that will be modelled as Al203

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
            >>> material_Al2O3_UDM(1, wavelength_range="visible") -> layer 1 will be Al2O3 from the UDM data set good from 0.01 to 100 eV
            """
            # get path to the Al203 data file
            file_path = path + "data/Al2O3_udm.txt"
            # now read Tin data into a numpy array
            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index
            e_data = file_data[:, 0]
            wl_nm = 1239.84193 / e_data

            wl_si = np.flip(wl_nm) * 1e-9

            n_array = np.flip(file_data[:,1])
            k_array = np.flip(file_data[:,2])

            n_spline = InterpolatedUnivariateSpline(
                wl_si, n_array, k=1
            )
            k_spline = InterpolatedUnivariateSpline(
                wl_si, k_array, k=1
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

            # now read Ag data into a numpy array
            file_data = np.loadtxt(file_path)
            # file_path[:,0] -> wavelengths in meters
            # file_path[:,1] -> real part of the refractive index
            # file_path[:,2] -> imaginary part of the refractive index

            # sometimes there are duplicate wavelength, n, and k entries
            # in a data set; we want only the unique elements
            idx = self._find_unique_ri_file_data(file_data[:, 0])

            n_spline = InterpolatedUnivariateSpline(
                file_data[idx, 0], file_data[idx, 1], k=1
            )
            k_spline = InterpolatedUnivariateSpline(
                file_data[idx, 0], file_data[idx, 2], k=1
            )

            self._refractive_index_array[:, layer_number] = n_spline(
                self.wavelength_array
            ) + 1j * k_spline(self.wavelength_array)

    def material_Pb(self, layer_number, wavelength_range="visible", override="true"):
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
            >>> material_pb(1, wavelength_range="visible") -> layer 1 will be Pb from the Werner data set good from visible to 2.47 microns
            >>> material_pb(2, wavelength_range="ir") -> layer 2 will be Pb from the ordal data set good until 667 microns
            """

            # dictionary specific to W with wavelength range information corresponding to different
            # data sets
            data1 = {
                "file": "data/Pb_Werner.txt",
                "lower_wavelength": 1.758600000e-08,
                "upper_wavelength": 2.479684000e-06,
            }
            data2 = {
                "file": "data/Pb_Ordal.txt",
                "lower_wavelength": 0.00000066700000,
                "upper_wavelength": 0.00066700000000,
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
                    file_path = path + "data/Pb_Werner.txt"

                elif wavelength_range == "ir" or wavelength_range == "long":
                    file_path = path + "data/Pb_Ordal.txt"

            # now read Pb data into a numpy array
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

    def _read_CIE(self):
        """Reads CIE data and stores as attributes self.cie_cr, self.cie_cg, self.cie_cb

        Arguments
        ----------
        None

        References
        ----------
        Equations 29-32 of https://github.com/FoleyLab/wptherml/blob/master/docs/Equations.pdf

        Attributes
        ----------
        _cie_cr : 1 x wavelength array of floats
            data corresponding to red cone response function in integrand of Eq. 29

        _cie_cg : 1 x wavelength array of floats
            data corresponding to the green cone response function in integrand of Eq. 30

        _cie_cb : 1 x wavelength array of floats
            data corresponding to the blue cone response function in integrand of Eq. 31

        Returns
        -------
        None
        """
        # initialize cie arrays?
        self._cie_cr = np.zeros_like(self.wavelength_array)
        self._cie_cg = np.zeros_like(self.wavelength_array)
        self._cie_cb = np.zeros_like(self.wavelength_array)
        # )
        # get path to the cie data
        file_path = path + "data/cie_cmf.txt"
        # now read Rh data into a numpy array
        file_data = np.loadtxt(file_path)
        # file_data[:,0] -> wavelengths in nm
        # file_data[:,1] -> cr response function
        # file_data[:,2] -> cg response function
        # file_data[:,3] -> cb resposne function

        _cr_spline = InterpolatedUnivariateSpline(
            file_data[:, 0] * 1e-9, file_data[:, 1], k=1
        )
        _cg_spline = InterpolatedUnivariateSpline(
            file_data[:, 0] * 1e-9, file_data[:, 2], k=1
        )
        _cb_spline = InterpolatedUnivariateSpline(
            file_data[:, 0] * 1e-9, file_data[:, 3], k=1
        )
        # values of data file at 500 nm
        expected_values = np.array([0.0049, 0.3230, 0.2720])
        spline_values = np.array(
            [_cr_spline(500e-9), _cg_spline(500e-9), _cb_spline(500e-9)]
        )
        assert np.allclose(expected_values, spline_values)
        self._cie_cr[:] = _cr_spline(self.wavelength_array)
        self._cie_cg[:] = _cg_spline(self.wavelength_array)
        self._cie_cb[:] = _cb_spline(self.wavelength_array)

    def _read_AM(self):
        """Reads AM1.5 data and returns an array of the AM1.5 data evaluated at each value of
            self.wavelength_array

        Arguments
        ----------
        None

        References
        ----------
        add

        Attributes
        ----------
        None

        Returns
        -------
        1 x number_of_wavelengths array of floats
            Data corresponding the AM1.5 spectrum evaluated at each value of self.wavelength_array

        """

        # get path to the AM data
        file_path = path + "data/scaled_AM_1_5.txt"
        # now read Rh data into a numpy array
        file_data = np.loadtxt(file_path)
        # file_data[:,0] -> wavelengths in m
        # file_data[:,1] -> solar spectrum in W / m / m^2 / sr

        _solar_spline = InterpolatedUnivariateSpline(
            file_data[:, 0], file_data[:, 1], k=1
        )

        # values of data file at 615 nm
        # 0.000000615000000       1325400000.0000000000000000000000
        _expected_value = 1325400000.0
        _spline_value = _solar_spline(615e-9)
        assert np.isclose(_expected_value, _spline_value)
        return _solar_spline(self.wavelength_array)

    def _read_Atmospheric_Transmissivity(self):
        """Reads atmospherical transmissivity data and returns
            an array of this data evaluated at each value of self.wavelength_array

        Arguments
        ----------
        None

        References
        ----------
        add

        Attributes
        ----------
        None

        Returns
        -------
        1 x number_of_wavelengths array of floats
            atmospheric transmissivity evaluated at each value of self.wavelength_array
        """

        # get path to the AM data
        file_path = path + "data/Atmospheric_transmissivity.txt"
        # now read Rh data into a numpy array
        file_data = np.loadtxt(file_path)
        # file_data[:,0] -> wavelengths in m
        # file_data[:,1] -> atmospheric transmissivity

        # get indices of unique elements
        idx = self._find_unique_ri_file_data(file_data[:, 0])

        _atrans_spline = InterpolatedUnivariateSpline(
            file_data[idx, 0], file_data[idx, 1], k=1
        )

        # values of data file at 7.1034e-06 meters (7.1034 microns) -> T = 0.561289
        _expected_value = 0.561289
        _spline_value = _atrans_spline(7.1034e-6)
        assert np.isclose(_expected_value, _spline_value)
        return _atrans_spline(self.wavelength_array)
    
    def _EQE_spectral_response(self):
        """ 
        Will compute the spectral response function using tabulated EQE values for user input thickness
            from 

            "Optical Properties and Modeling of 2D Perovskite Solar Cells",
            Bin Liu, Chan Myae Myae Soe, Constantinos C. Stoumpos, Wanyi Nie, Hsinhan Tsai, Kimin
            Lim, Aditya D. Mohite, Mercouri G. Kanatzidis, Tobin J. Marks, Kenneth D. Singer
            Advanced Materials, (34), 1. January 6, 2022 
            https://doi.org/10.1002/adma.202107211

            on EQE curves of Pb5 perovskite-based devices.

            Using formula SR = q * EQE * \lambda / (h * c)

        Attributes
        ----------
        psc_thickness : 
                        User-input thickness in nm of PSC.
        _eqe_array :
                        Table of EQE values extrapolated from graph.
        _sr_array : 
                        Calculated spectral response based on formula, wavelength, tabulated values, and constants.
        _eqe_spline :
                        Spline best fit based on wavelength array and EQE data.
        _sr_spline :
                        Spline best fit based on SR array and wavelength array.

        Returns
        -------
        None
        
        """

        # Initialize wavelength array and variable for thickness
        _wavelength_array = np.array([250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850]) * 1e-9

        # Check for psc values
        if self.psc_thickness_option == 110:
            _eqe_array = np.array([0, 0, 22, 46, 58, 59, 54, 40, 24, 18, 4,	0, 0])

        elif self.psc_thickness_option == 200:
            _eqe_array = np.array([0, 0, 25, 52, 72, 74, 71, 59, 39, 22, 5, 0, 0])

        elif self.psc_thickness_option == 250:
            _eqe_array = np.array([0, 0, 20, 40, 54, 59, 58, 48, 39, 32, 14, 0,	0])
        
        elif self.psc_thickness_option == 410:
            _eqe_array = np.array([0, 0, 16, 27, 33, 36, 35, 32, 31, 24, 7, 0, 0])
                    
        else:
            _eqe_array = np.array([0, 0, 25, 52, 72, 74, 71, 59, 39, 22, 5, 0, 0])

        # Transferring from percentages and SR calculation
        _eqe_array = _eqe_array * 0.01
        _sr_array = constants.e * _eqe_array * _wavelength_array / (constants.h * constants.c)

        # Spline for line of best fit based on current data
        _eqe_spline = InterpolatedUnivariateSpline(
            _wavelength_array, _eqe_array, k=1
        )

        _sr_spline = InterpolatedUnivariateSpline(
                _wavelength_array, _sr_array, k=1
            )
        
        self.perovskite_eqe = _eqe_spline(self.wavelength_array)
        self.perovskite_spectral_response = _sr_spline(self.wavelength_array)
    

