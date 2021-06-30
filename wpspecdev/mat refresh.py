    def material_Mat_name(self, layer_number, wavelength_range="visible", override="true"):
        if layer_number > 0 and layer_number < (self.number_of_layers - 1):
            """ defines the refractive index of layer layer_number to be !Mat!
            
            Arguments
            ----------
            layer_number : int
            specifies the layer of the stack that will be modelled as !Mat!
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
            >>> material_!Mat!(1, wavelength_range="visible") -> layer 1 will be !Mat! from the Rodriguez data set good from visible to 1.5 microns
            >>> material_!Mat!(2, wavelength_range="ir") -> layer 2 will be !Mat! from the Bright data set good until 1000 microns
            """

            # dictionary specific to Ta2O5 with wavelength range information corresponding to different
            # data sets
            data1 = {
                "file": "data/file_name",
                "lower_wavelength": 0,
                "upper_wavelength": 0
            }
            data2 = {
                "file": "data/file_name",
                "lower_wavelength": 0,
                "upper_wavelength": 0
            }

            shortest_wavelength = self.wavelength_array[0]
            longest_wavelength = self.wavelength_array[self.number_of_wavelengths-1]

            

            if shortest_wavelength >= data1["lower_wavelength"] and longest_wavelength <= data1["upper_wavelength"]:
                file_path = path + data1["file"]
            
            elif shortest_wavelength >= data2["lower_wavelength"] and longest_wavelength <= data2["upper_wavelength"]:
                file_path = path + data2["file"]
            
            else:
                file_path = path + data1["file"]
             

            if override=='false':
                # make sure the wavelength_range string is all  lowercase
                wavelength_range = wavelength_range.lower()
                if wavelength_range=="visible" or wavelength_range=="short" or wavelength_range=="vis":
                    file_path = path + "data/file_name"
                    
                elif wavelength_range=="ir" or wavelength_range=="long":
                    file_path = path + "data/file_name"

            


            print("read from ",file_path)
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






def test_material_Mat_name():
    """ Dictionaries from material_Au """
    data1 = {
        "file": "data/file_name",
        "lower_wavelength": 0,
        "upper_wavelength":0,
        "test_wavelength": 0,
        "test_n":      0,
        "test_k": 0     
    }
    data2 = {
        "file": "data/file_name",
        "lower_wavelength": 0,
        "upper_wavelength":0,
        "test_wavelength": 0,
        "test_n": 0,
        "test_k": 0

    }
   
 

    
    expected_n_1 = data1["test_n"]
    expected_k_1 = data1["test_k"]
    wavelength_1 = data1["test_wavelength"]

    expected_n_2 = data2["test_n"]
    expected_k_2 = data2["test_k"]
    wavelength_2 = data2["test_wavelength"]

    # create test multilayer for data1
    material_test._create_test_multilayer(central_wavelength=wavelength_1)
    # define central layer as Ta2O5 using data1
    material_test.material_Mat_name(1)""

    result_n_1 = np.real(material_test._refractive_index_array[1, 1])
    result_k_1 = np.imag(material_test._refractive_index_array[1, 1])

    # update test multilayer for data2
    material_test._create_test_multilayer(central_wavelength=wavelength_2)
    # define central layer as Ta2O5 using data2
    material_test.material_Mat_name(1) ""

    result_n_2 = np.real(material_test._refractive_index_array[1, 1])
    result_k_2 = np.imag(material_test._refractive_index_array[1, 1])


    assert np.isclose(result_n_1, expected_n_1, 1e-3)
    assert np.isclose(result_k_1, expected_k_1, 1e-3)
    assert np.isclose(result_n_2, expected_n_2, 1e-3)
    assert np.isclose(result_k_2, expected_k_2, 1e-3)