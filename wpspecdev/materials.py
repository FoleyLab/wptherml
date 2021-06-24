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
            # get path to the sio2 data file
            file_path = path + "data/SiO2_ir.txt"
            # now read sio2 data into a numpy array
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
            # get path to the tio2 data file
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

    def material_Ta2O5(self, layer_number):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            # get path to the tio2 data file
            file_path = path + "data/Ta2O5_Bright.txt"
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

    def material_TiN(self, layer_number):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            # get path to the tio2 data file
            file_path = path + "data/TiN_ellipsometry_data.txt"
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

    def material_static_refractive_index(self, layer_number, refractive_index):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            self._refractive_index_array[:, layer_number] = (
                np.ones(len(self.wavelength_array), dtype=complex) * refractive_index
            )

    def material_Al(self, layer_number):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            # get path to the Al data file
            file_path = path + "data/Al.txt"
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
            # get path to the W data file
            file_path = path + "data/W.txt"
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
            self._refractive_index_array[:, layer_number] = (
                np.ones(len(self.wavelength_array), dtype=complex) * 2.4
            )
            # get path to the Platinum data file
            file_path = path + "data/Pt.txt"
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
            # get path to the AlN data file
            file_path = path + "data/AlN.txt"
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
            self._refractive_index_array[:, layer_number] = (
                np.ones(len(self.wavelength_array), dtype=complex) * 2.4
            )
            # get path to the Lead data file
            file_path = path + "data/Pb.txt"
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
            self._refractive_index_array[:, layer_number] = (
                np.ones(len(self.wavelength_array), dtype=complex) * 2.4
            )
            # get path to the HfO2 data file
            file_path = path + "data/HfO2.txt"

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
            self._refractive_index_array[:, layer_number] = (
                np.ones(len(self.wavelength_array), dtype=complex) * 2.4
            )
            # get path to the Re data file
            file_path = path + "data/Re.txt"
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

    def material_Au(self, layer_number):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            self._refractive_index_array[:, layer_number] = (
                np.ones(len(self.wavelength_array), dtype=complex) * 2.4
            )
            # get path to the Au data file
            file_path = path + "data/Au_ri.txt"
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
            self._refractive_index_array[:, layer_number] = (
                np.ones(len(self.wavelength_array), dtype=complex) * 2.4
            )
            # get path to the Rh data file
            file_path = path + "data/Rh.txt"
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

    def material_Si(self, layer_number):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            self._refractive_index_array[:, layer_number] = (
                np.ones(len(self.wavelength_array), dtype=complex) * 2.4
            )
            # get path to the Si data file
            file_path = path + "data/Si.txt"
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
