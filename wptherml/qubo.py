
    class _qubo_thickness_and_alloy_optimization:
        def __init__(self, outer_instance, optimization_dict: dict, fom_func):
            """
            pass a test_args dicticionary like
            test_args = {
                "wavelength_list": [300e-9, 6000e-9, 1000],
                "Material_List": ["Air", "Al2O3", "SiO2", "TiO2", "SiO2", "Al2O3", "W", "Air"],
                "Thickness_List": [0, 20e-9, 255e-9, 150e-9, 255e-9, 10e-9, 900e-9, 0],
                "temperature": 1700,
                "therml": True
            }

            pass another dictionary to specify which parameters to optimize
            thickness_adjustment_dict is a dictionary of indices of the materials_list to adjust thickness along with a scaling factor to explore a range of thicknesses


            explanation:
            optimization_dict = {
                "thickness_adjustment_dict": {indice_to_adjust_thickness: scaling_factor_of_original_thickness},
                "thickness_bit_length": {indice_to_adjust_thickness: number_of_bits_to_represent_thickness},
                "alloy_dict": {indice_of_alloy: [alloy_component_number_1(same as alloy indice), alloy_componenet_2]},
                "alloy_bit_length": {indice_of_alloy: number_of_bits_to represent alloy volume fraction}
            }

            example:
            optimization_dict = {
                "thickness_adjustment_dict": {1: 2},
                "thickness_bit_length": {1: 6},
                "alloy_dict": {1: [1,6]},
                "alloy_bit_length": {1: 6}
            }
            """

            # instance of outer class Opt Driver object
            self.outer_instance = outer_instance
            self.fom_func = fom_func

            self.test_args = self.outer_instance.args
            self.stack_height = len(self.test_args["Material_List"])
            self.optimization_dict = optimization_dict

            # store a copy of original test_args
            self.og_test_args = copy.deepcopy(self.outer_instance.args)
            self.og_test = copy.deepcopy(self.outer_instance)

            # get and store which indices of materials list to adjust thicknesses for
            self.thickness_adjustment_dict = optimization_dict[
                "thickness_adjustment_dict"
            ]
            # also store how many bits to represent each
            self.thickness_bit_lengths = optimization_dict["thickness_bit_length"]
            self.thickness_list = self.test_args["Thickness_List"]

            # need to get and store refractive indexes for these alloys for maxwell garnett
            # get and store indice of place where alloy will be and what the alloy make up will be
            self.alloy_dict = optimization_dict["alloy_dict"]
            # also need to store how many bits represent the alloy
            self.alloy_bit_lengths = optimization_dict["alloy_bit_length"]

            self.num_bits = 0
            # calculate total number of bits needed
            for key in self.thickness_bit_lengths:
                self.num_bits += self.thickness_bit_lengths[key]
            for key in self.alloy_bit_lengths:
                self.num_bits += self.alloy_bit_lengths[key]

            # breaking up bitstring by indice say for indice 1 you want to optimize alloy and thickness
            # it will be first bits for alloy then next bits for thickness and continue by indice
            # create a dictionary to describe how bitstring will be split up for update_multilayer function
            # of form {1: {'A': 6, 'T': 6}}
            self.bit_split_dict = {}
            for i in range(0, self.stack_height):
                if i in self.alloy_bit_lengths or i in self.thickness_bit_lengths:
                    self.bit_split_dict[i] = {}
                    if i in self.alloy_bit_lengths:
                        self.bit_split_dict[i]["A"] = self.alloy_bit_lengths[i]
                    if i in self.thickness_bit_lengths:
                        self.bit_split_dict[i]["T"] = self.thickness_bit_lengths[i]
            # print(self.bit_split_dict)

            # create a dictionary to store original ranges for the optimizaiton
            # of the form {'A': [{1: [0, 1]}], 'T': [{1: [0, 4e-08]}, {5: [0, 2e-08]}]}
            self.ranges = {"A": [], "T": []}
            for key in self.bit_split_dict:
                # key in this case is an indice
                if "A" in self.bit_split_dict[key]:
                    self.ranges["A"].append({key: [0, 1]})
                if "T" in self.bit_split_dict[key]:
                    self.ranges["T"].append(
                        {
                            key: [
                                0,
                                self.thickness_adjustment_dict[key]
                                * self.thickness_list[key],
                            ]
                        }
                    )

            print(self.ranges)

        def random_binary(self, length, num):
            # generate a 2d array of random binary string
            # specify length of binary strings as length and number you need as num
            arr = []
            for i in range(0, num):
                arr.append(list(np.random.choice([0, 1], size=(length,))))
            return arr

        def binary_to_int(self, bin):
            # convert binary to an integer
            num = int("".join(str(x) for x in bin), 2)
            return num

        def MaxwellGarnett(self, ri_1, ri_2, fraction):
            """a function that will compute the alloy refractive
            index between material_1 and material_2 using
            Maxwell-Garnett theory."""
            # define _eps_d as ri_1 ** 2
            _eps_d = ri_1 * ri_1

            # define _eps_m as ri_2 * ri_2
            _eps_m = ri_2 * ri_2

            # numerator of the Maxwell-Garnett model
            _numerator = _eps_d * (
                2 * fraction * (_eps_m - _eps_d) + _eps_m + 2 * _eps_d
            )
            # denominator of the Maxwell-Garnett model
            _denominator = 2 * _eps_d + _eps_m + fraction * (_eps_d - _eps_m)

            # _numerator / _denominator is epsilon_effective, and se want n_eff = sqrt(epsilon_eff)
            return np.sqrt(_numerator / _denominator)

        def learning_loop(
            self,
            num_iterations=1,
            num_to_train=400,
            plot_train_data=True,
            reduction_factor=2,
            n_epochs=60,
            l2_lambda=0.0001,
            l1_lambda=0,
            K=5,
            LR=0.05,
        ):
            # generate set of random bitstrings to use
            bitstrings = self.random_binary(self.num_bits, num_to_train)

            # construct a data frame with the relavent information from the random bitstrings
            """
            store data like

            bitstrings | alloy_indice_1 | thickness_indice_1 | FOM

            0001001   |    10e-9           | 0.2            | FOM
            
            """
            df = self.update_multilayer(bitstrings)

            for i in range(0, num_iterations):
                N = len((df.loc[:, "bitstrings"]).tolist()[0])
                K = K
                fm = self.outer_instance._FMTrain(N, K)
                fm.train(
                    np.array(df.loc[:, "bitstrings"]).tolist(),
                    np.array(df.loc[:, "FOM"]).reshape(-1, 1),
                    split_size=0.8,
                    batch_size=10000,
                    n_epochs=n_epochs,
                    minmaxscale=True,
                    LR=LR,
                    l2_lambda=l2_lambda,
                    l1_lambda=l1_lambda,
                    opt="LBFGS",
                )
                best_sample, samples_list = fm.get_qubo_hamiltonian_minimas(100)

                qubo_result_df = self.convert_qubo_result_to_df(best_sample.sample)
                samples_list_df = self.convert_qubo_results_to_df(samples_list)

                # add new ones to df from qubo minima
                df = pd.concat([df, samples_list_df], ignore_index=True, sort=False)

                # want to weight the lower values higher so that model learns lower regions better
                # duplicate bottom 15% of data, the 15% of data with lowest FOM, and add back to training data
                if i % 200 == 0 or i == 0:
                    df_copy = copy.deepcopy(df.sort_values(by=["FOM"]))

                    bottom_25 = int(df_copy.shape[0] * 0.15)
                    df_copy = pd.DataFrame(df_copy.iloc[0:bottom_25, :])

                    df = pd.concat([df, df_copy])

                # redo ranges every 400 iterations
                if i != 0 and i % 400 == 0:
                    qubo_result_df = df[df.FOM == df.FOM.min()]
                    qubo_result_df = qubo_result_df.reset_index()

                    # reduce ranges to
                    for j in range(0, len(self.thickness_list)):
                        # for the alloy portion'
                        for l in range(0, len(self.ranges["A"])):
                            if j in self.ranges["A"][l]:
                                center_of_new_range = qubo_result_df.loc[
                                    0, str("alloy_indice_" + str(j))
                                ]

                                new_range = (
                                    self.ranges["A"][l][j][1]
                                    - self.ranges["A"][l][j][0]
                                ) / reduction_factor

                                print(min(1, 2))
                                print(max(1, 2))
                                self.ranges["A"][l][j][0] = min(
                                    max(0, (center_of_new_range - (new_range / 2))), 1
                                )
                                self.ranges["A"][l][j][1] = min(
                                    max(0, (center_of_new_range + (new_range / 2))), 1
                                )

                        # for the thickness portion
                        for l in range(0, len(self.ranges["T"])):
                            if j in self.ranges["T"][l]:
                                center_of_new_range = qubo_result_df.loc[
                                    0, str("thickness_indice_" + str(j))
                                ]
                                new_range = (
                                    self.ranges["T"][l][j][1]
                                    - self.ranges["T"][l][j][0]
                                ) / reduction_factor

                                self.ranges["T"][l][j][0] = min(
                                    max(0, (center_of_new_range - (new_range / 2))), 1
                                )
                                self.ranges["T"][l][j][1] = min(
                                    max(0, (center_of_new_range + (new_range / 2))), 1
                                )

                        # remake data after resetting ranges

                    bitstrings = self.random_binary(self.num_bits, num_to_train)

                    # construct a data frame with the relavent information from the random bitstrings
                    """
                    store data like

                    bitstrings | alloy_indice_1 | thickness_indice_1 | FOM

                    0001001   |    10e-9           | 0.2            | FOM
                    
                    """
                    df = self.update_multilayer(bitstrings)

            return qubo_result_df.loc[:, "FOM"]

        def update_multilayer(self, bitstrings):
            """function to compute the FOM from bitstring

            Arguments
            ---------
            bistrings: dataframe with a lot of random bitstrings stored as lists

            Returns
            -------
            dataframe with FOM for that bitstring and the values represented by the bitstring
            """

            # creaating arrangement of dataframe
            columns = ["bitstrings"]

            for key in self.bit_split_dict:
                if "A" in self.bit_split_dict[key]:
                    columns.append("alloy_indice_" + str(key))

                if "T" in self.bit_split_dict[key]:
                    columns.append("thickness_indice_" + str(key))

            columns.append("FOM")

            df = pd.DataFrame(columns=columns)

            for i in range(0, len(bitstrings)):
                # store the bitstrikng in the bit strings column

                new_row_for_df = []

                current_bit = 0

                new_row_for_df.append(bitstrings[i])

                # loop through indices for optimization
                for indice in self.bit_split_dict:
                    # A or T or both for that indice
                    for key2 in self.bit_split_dict[indice]:
                        # bitstring for current split
                        bitstring_seg = bitstrings[i][
                            current_bit : current_bit
                            + self.bit_split_dict[indice][key2]
                        ]

                        current_bit = current_bit + self.bit_split_dict[indice][key2]

                        if key2 == "A":
                            # loop through the x array
                            # w = (x[0] * 2 ** 3 + x[1] * 2 ** 2 + x[2] * 2 ** 1 + x[3] * 2 ** 0) / 15
                            x = np.flip(bitstring_seg)
                            length = len(x)
                            max_x = 2 ** len(x)
                            fw = 0

                            for j in range(length):
                                fw += 2**j * x[j]

                            for k in range(0, len(self.ranges["A"])):
                                if indice in self.ranges["A"][k]:
                                    range_x = self.ranges["A"][k][indice]

                            fw = range_x[0] + fw / max_x * (range_x[1] - range_x[0])

                            # add to df
                            new_row_for_df.append(fw)

                            _n1 = np.copy(
                                self.og_test._refractive_index_array[
                                    :, self.alloy_dict[indice][0]
                                ]
                            )
                            _n2 = np.copy(
                                self.og_test._refractive_index_array[
                                    :, self.alloy_dict[indice][1]
                                ]
                            )

                            n_eff = self.MaxwellGarnett(_n1, _n2, fw)
                            self.outer_instance._refractive_index_array[
                                :, self.alloy_dict[indice][0]
                            ] = n_eff

                        if key2 == "T":
                            y = np.flip(bitstring_seg)
                            length_y = len(y)
                            max_y = 2 ** len(y)
                            val = 0

                            scaling_factor = self.thickness_adjustment_dict[indice]
                            hundred_percent = max_y / scaling_factor

                            for j in range(length_y):
                                val += 2**j * y[j]

                            percent = (val) / hundred_percent

                            for k in range(0, len(self.ranges["T"])):
                                if indice in self.ranges["T"][k]:
                                    range_y = self.ranges["T"][k][indice]

                            self.test_args["Thickness_List"][indice] = range_y[0] + (
                                self.og_test_args["Thickness_List"][indice] * percent
                            ) * (range_y[1] - range_y[0]) / (
                                self.og_test_args["Thickness_List"][indice]
                                * scaling_factor
                            )
                            self.outer_instance.thickness_array[indice] = range_y[0] + (
                                self.og_test_args["Thickness_List"][indice] * percent
                            ) * (range_y[1] - range_y[0]) / (
                                self.og_test_args["Thickness_List"][indice]
                                * scaling_factor
                            )

                            new_row_for_df.append(
                                self.test_args["Thickness_List"][indice]
                            )

                fom = self.fom_func(self.outer_instance)
                new_row_for_df.append(fom)
                df.loc[len(df.index)] = new_row_for_df

            return df

        def convert_qubo_result_to_df(self, best_sample):
            print("best_sample: ", best_sample)
            new_dict = {}
            binary_list = []

            for key in best_sample:
                temp_key = int(key[1:])
                temp_value = best_sample[key]

                new_dict[temp_key] = temp_value

            best_sample = new_dict

            myKeys = list(best_sample.keys())
            myKeys.sort()
            sorted_dict = {i: best_sample[i] for i in myKeys}

            for key in sorted_dict:
                binary_list.append(sorted_dict[key])

            minima_df = self.update_multilayer([np.array(binary_list)])

            print(minima_df)

            return minima_df

        def convert_qubo_results_to_df(self, samples_list):
            samples = []
            for m in samples_list:
                q = m.sample

                new_dict = {}
                binary_list = []

                for key in q:
                    temp_key = int(key[1:])
                    temp_value = q[key]

                    new_dict[temp_key] = temp_value

                q = new_dict

                myKeys = list(q.keys())
                myKeys.sort()
                sorted_dict = {i: q[i] for i in myKeys}

                for key in sorted_dict:
                    binary_list.append(sorted_dict[key])

                samples.append(binary_list)

            samples = [list(tupl) for tupl in {tuple(item) for item in samples}]

            minima_df = self.update_multilayer(samples)

            # print(minima_df)

            return minima_df

    class _qubo_combinatorial_structure_optimization:
        def __init__(self, outer_instance, optimization_dict, fom_func):
            # instance of outer class Opt Driver object
            self.outer_instance = outer_instance

            self.optimization_dict = optimization_dict

            self.fom_func = fom_func

            # optimizaiton dictionary with form of
            """
            for the lists of materials, the number of materials must be a power of 2, so 2, 4, 8
            layers also need to be in order no gaps unless you go to the next line so you can't have (1,2,3,5) you would need 5 on its own line below
            {
                (1,2,3,4): ("SiO2", "HfO2", "AlN", "TiO2"),
                (5,6,7,8): ("SiO2", "HfO2", "TiO2", "Al2O3")
            }
            """

            # need to parse this dictionary and construct a list of methods that hold the materials
            # to be able to modify cool_ml object
            # list of list of methods, new list for each set of layers to modify

            self.material_methods = []

            self.num_bits = 0

            for key in optimization_dict:
                materials_methods_inner = []

                materials = list(optimization_dict[key])

                # store number of bits that are needed to represent the material
                for i in range(0, len(key)):
                    self.num_bits += int(np.log((len(self.optimization_dict[key])), 2))

                # need to store methods to switch materials in structure and evaluate new structures
                for material in materials:
                    if material == "SiO2":
                        materials_methods_inner.append(
                            self.outer_instance.material_SiO2
                        )
                    elif material == "HfO2":
                        materials_methods_inner.append(
                            self.outer_instance.material_HfO2
                        )
                    elif material == "AlN":
                        materials_methods_inner.append(self.outer_instance.material_AlN)
                    elif material == "Al2O3":
                        materials_methods_inner.append(
                            self.outer_instance.material_Al2O3
                        )
                    elif material == "TiO2":
                        materials_methods_inner.append(
                            self.outer_instance.material_TiO2
                        )
                    elif material == "TiN":
                        materials_methods_inner.append(self.outer_instance.material_TiN)
                    elif material == "ZrO2":
                        materials_methods_inner.append(
                            self.outer_instance.material_ZrO2
                        )
                    elif material == "Si3N4":
                        materials_methods_inner.append(
                            self.outer_instance.material_Si3N4
                        )

                self.material_methods.append(materials_methods_inner)

            self.materials = []

            # store a list with strings of material names so that bitstrings can be mapped back to structures
            for key in optimization_dict:
                materials_inner = []

                materials = list(optimization_dict[key])

                for material in materials:
                    if material == "SiO2":
                        materials_inner.append("SiO2")
                    elif material == "HfO2":
                        materials_inner.append("HfO2")
                    elif material == "AlN":
                        materials_inner.append("AlN")
                    elif material == "Al2O3":
                        materials_inner.append("Al2O3")
                    elif material == "TiO2":
                        materials_inner.append("TiO2")
                    elif material == "TiN":
                        materials_inner.append("TiN")
                    elif material == "ZrO2":
                        materials_inner.append("ZrO2")
                    elif material == "Si3N4":
                        materials_inner.append("Si3N4")

                self.materials.append(materials_inner)

        def random_binary(self, length, num):
            # generate a 2d array of random binary string
            # specify length of binary strings as length and number you need as num
            arr = []
            for i in range(0, num):
                arr.append(list(np.random.choice([0, 1], size=(length,))))
            return arr

        def learning_loop(
            self,
            num_iterations=10,
            num_to_train=100,
            n_epochs=60,
            l2_lambda=0.0001,
            l1_lambda=0,
            K=5,
            LR=0.05,
        ):
            # generate set of random bitstrings to use
            bitstrings = self.random_binary(self.num_bits, num_to_train)

            # construct a data frame with the relavent information from the random bitstrings
            """
            store data like

            bitstrings |       materials_list               | FOM

            0001001    | ['SiO2', 'AlN', 'TiO2', 'Al2O3']   | FOM
            
            """

            # going to store all data generated in dataframe
            df = pd.DataFrame(columns=["bitstrings", "materials_list", "FOM"])

            # evaluate bitstrings to get figure of merit, FOM to create training data
            for j in range(0, len(bitstrings)):
                bitstring = bitstrings[j]
                data_from_bitstring = self.update_multilayer(bitstring)

                data_from_bitstring.insert(0, bitstring)

                df.loc[len(df.index)] = data_from_bitstring

            # loop where factorization machine is trained, minima is predicted, evaluated, and added to training data
            for i in range(0, num_iterations):
                N = len((df.loc[:, "bitstrings"]).tolist()[0])
                K = K
                fm = self.outer_instance._FMTrain(N, K)
                fm.train(
                    np.array(df.loc[:, "bitstrings"]).tolist(),
                    np.array(df.loc[:, "FOM"]).reshape(-1, 1),
                    split_size=0.8,
                    batch_size=10000,
                    n_epochs=n_epochs,
                    minmaxscale=True,
                    LR=LR,
                    l2_lambda=l2_lambda,
                    l1_lambda=l1_lambda,
                    opt="LBFGS",
                )
                best_sample = fm.get_qubo_hamiltonian_minima()

                qubo_result_list = self.convert_qubo_result_to_df(best_sample.sample)

                print(qubo_result_list)

                df.loc[len(df.index)] = qubo_result_list

        def update_multilayer(self, x):
            """function to assign a multilayer based on an input bitstring x

            Arguments
            ---------
            x : bitstring
                bitstring representation of the structure

            Returns
            -------
            net_cooling_power : float
                the figure of merit, also stored as an attribule
            """

            materials_list = []

            current_bit = 0

            i = 0
            for key in self.optimization_dict:
                bit_length = int(np.log((len(self.optimization_dict[key])), 2))

                for layer in key:
                    bits = x[current_bit : current_bit + bit_length]
                    bits = "".join(str(bit) for bit in bits)
                    num = int(bits, 2)

                    self.material_methods[i][num](layer)

                    materials_list.append(self.materials[i][num])

                    current_bit = current_bit + bit_length

                i += 1

            fom = self.fom_func(self.outer_instance)

            return [materials_list, fom]

        def convert_qubo_result_to_df(self, best_sample):
            """
            take qubo hamiltonian minima and convert result back to multilayer and place date in dataframe
            """

            new_dict = {}
            binary_list = []

            for key in best_sample:
                temp_key = int(key[1:])
                temp_value = best_sample[key]

                new_dict[temp_key] = temp_value

            best_sample = new_dict

            myKeys = list(best_sample.keys())
            myKeys.sort()
            sorted_dict = {i: best_sample[i] for i in myKeys}

            for key in sorted_dict:
                binary_list.append(sorted_dict[key])

            minima_list = self.update_multilayer(binary_list)

            minima_list.insert(0, binary_list)

            return minima_list

    class _FMTrain:
        class _TorchFM(nn.Module):
            def __init__(self, n=None, k=None):
                super().__init__()
                self.V = nn.Parameter(
                    torch.FloatTensor(n, k).uniform_(-0.1, 0.1), requires_grad=True
                )

                self.lin = nn.Linear(n, 1)

                self.n = n
                self.k = k

            def forward(self, x):
                out_1 = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True)  # S_1^2
                out_2 = torch.matmul(x.pow(2), self.V.pow(2)).sum(
                    1, keepdim=True
                )  # S_2

                out_inter = 0.5 * (out_1 - out_2)

                out_lin = self.lin(x)

                out = out_inter + out_lin

                return out

            def return_params(self):
                V = self.V.detach().cpu().numpy().copy()

                return self.lin, V

        def __init__(self, n, k, verbose=False):
            # Define the model
            self.model = self._TorchFM(n, k)
            self.n = n
            self.k = k
            self.verbose = verbose

        def train(
            self,
            X,
            y,
            split_size=0.7,
            batch_size=1,
            n_epochs=1,
            minmaxscale=True,
            standardscale=False,
            opt="LBFGS",
            LR=0.01,
            l2_lambda=0.001,
            l1_lambda=0,
        ):
            """
            trains the factorization machine model, using data X and y

            """

            # train-test split for model evaluation
            (
                self.X_train_raw,
                self.X_test_raw,
                self.y_train_raw,
                self.y_test_raw,
            ) = train_test_split(X, y, train_size=split_size, shuffle=True)

            self.standardscale = standardscale
            self.minmaxscale = minmaxscale

            X_train = self.X_train_raw
            X_test = self.X_test_raw

            if self.standardscale or self.minmaxscale:
                # Standardizing data
                if self.standardscale:
                    self.scaler = StandardScaler()
                else:
                    self.scaler = MinMaxScaler((-1, 1))
                self.scaler.fit(self.y_train_raw)
                self.y_train = self.scaler.transform(self.y_train_raw)
                self.y_test = self.scaler.transform(self.y_test_raw)

            else:
                self.y_train = self.y_train_raw
                self.y_test = self.y_test_raw

            # Convert to 2D PyTorch tensors
            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(self.y_train, dtype=torch.float32).reshape(-1, 1)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            y_test = torch.tensor(self.y_test, dtype=torch.float32).reshape(-1, 1)

            # loss function and optimizer
            loss_fn = nn.MSELoss()  # mean square error
            # optimizer = optim.NAdam(model.parameters(), lr = .03)

            if opt == "LBFGS":
                optimizer = optim.LBFGS(self.model.parameters(), lr=LR)
            elif opt == "ADAM":
                optimizer = optim.Adam(self.model.parameters(), lr=LR)
            elif opt == "NADAM":
                optimizer = optim.NAdam(self.model.parameters(), lr=LR)
            elif opt == "SGD":
                optimizer = optim.SGD(self.model.parameters(), lr=LR)
            else:
                optimizer = optim.Adam(self.model.parameters(), lr=LR)

            batch_start = torch.arange(0, len(X_train), batch_size)

            # Hold the best model
            self.best_mse = np.inf  # init to infinity
            self.best_weights = None
            history = []
            history_train = []

            for epoch in range(n_epochs):
                running_loss = 0
                self.model.train()
                with tqdm.tqdm(
                    batch_start, unit="batch", mininterval=0, disable=True
                ) as bar:
                    if self.verbose:
                        bar.set_description(f"Epoch {epoch}")
                    for start in bar:
                        # take a batch
                        X_batch = torch.autograd.Variable(
                            X_train[start : start + batch_size], requires_grad=True
                        )
                        y_batch = torch.autograd.Variable(
                            y_train[start : start + batch_size], requires_grad=True
                        )

                        # loss closure function
                        def closure():
                            # Zero gradients
                            optimizer.zero_grad()
                            # Forward pass
                            y_pred = self.model(X_batch)
                            # Compute loss
                            loss = loss_fn(y_pred, y_batch)

                            # l2 regularization
                            l2_norm = sum(
                                p.pow(2.0).sum() for p in self.model.parameters()
                            )

                            l1_norm = sum(
                                torch.abs(p.pow(1.0)).sum()
                                for p in self.model.parameters()
                            )

                            loss = loss + (l2_lambda * l2_norm) + (l1_lambda * l1_norm)

                            # Backward pass
                            loss.backward()
                            return loss

                        # Update weights
                        optimizer.step(closure)
                        # Update the running loss
                        loss = closure()
                        running_loss += loss.item()
                        # print progress
                        if self.verbose:
                            bar.set_postfix(mse=float(loss))
                # evaluate accuracy at end of each epoch
                self.model.eval()
                y_pred_train = self.model(X_train)
                mse_train = loss_fn(y_pred_train, y_train)
                mse_train = float(mse_train)
                y_pred = self.model(X_test)
                mse = loss_fn(y_pred, y_test)
                mse = float(mse)
                if self.verbose:
                    print(
                        "epoch-",
                        epoch,
                        "   mse------",
                        mse,
                        "   mse_train------",
                        mse_train,
                    )
                history.append(np.log(mse))
                history_train.append(np.log(mse_train))
                if mse < self.best_mse:
                    self.best_mse = mse
                    self.best_weights = copy.deepcopy(self.model.state_dict())

            # restore model and return best accuracy
            self.model.load_state_dict(self.best_weights)

            if self.verbose:
                print("MSE: %.2f" % self.best_mse)
                print("RMSE: %.2f" % np.sqrt(self.best_mse))

        def eval(self, num_eval=20):
            # for testing the inference accuracy of the model after training

            self.model.eval()
            with torch.no_grad():
                # Test out inference with 5 samples from the original test set
                for i in range(num_eval):
                    X_sample = self.X_test_raw[i : i + 1]

                    if self.standardscale or self.minmaxscale:
                        y_pred = self.scaler.inverse_transform(y_pred)

                    X_sample = torch.tensor(X_sample, dtype=torch.float32)
                    y_pred = self.model(X_sample)
                    if self.standardscale or self.minmaxscale:
                        y_pred = self.scaler.inverse_transform(y_pred)

                    print(
                        f"{self.X_test_raw[i]} -> {y_pred[0]} (expected {self.y_test[i]})"
                    )

        def get_weights(self):
            # get weights from model as list
            model_params = copy.deepcopy(self.best_weights)
            return_list = [
                np.array(model_params["lin.bias"]),
                np.array(model_params["lin.weight"]),
                np.array(model_params["V"]),
            ]
            return return_list

        def return_weights(self):
            # get weights from model
            params = self.get_weights()
            V = np.array(np.array(params[2]).tolist())
            W = np.array(list(list(np.array(params[1]))[0]), dtype=np.float64)
            bias = np.array(params[0], dtype=np.float64)[0]

            return V, W, bias

        # following 2 methods
        # https://github.com/tsudalab/fmbqm/blob/master/fmbqm/factorization_machine.py

        def triu_mask(self, input_size, F=np.ndarray):
            # Generate a square matrix with its upper trianguler elements being 1 and others 0.

            mask = np.expand_dims(np.arange(input_size), axis=0)
            return (np.transpose(mask) < mask) * 1.0

        def VtoQ(self, V, F=np.ndarray):
            """Calculate interaction strength by inner product of feature vectors."""
            print(V.shape)
            # input_size = V.shape[1]
            input_size = V.shape[0]

            # Q = np.dot(np.transpose(V), V) # (d,d)
            Q = np.dot(V, np.transpose(V))  # (d,d)
            # Q =   V @ V.T
            triu = self.triu_mask(input_size)
            # print(triu)
            # return Q * np.triu(np.ones((input_size,input_size)), 1)
            return Q * triu

        def get_qubo_hamiltonian_minima(self):
            """convert factorization machine to qubo hamiltonian and solve the model, getting one result"""

            params = copy.deepcopy(self.get_weights())

            V = np.array(params[2], dtype=np.float64)

            W = np.array(np.array(params[1])[0], dtype=np.float64)

            bias = np.array(params[0], dtype=np.float64)[0]

            Q = V @ V.T

            Q = self.VtoQ(V)

            binary_len = int(self.n)
            qubo_basis = []
            for i in range(0, binary_len):
                new_string = "x" + str(i + 1)
                qubo_basis.append(Binary(new_string))

            # define qubo Hamiltonian
            H = 0
            for i in range(0, len(Q)):
                for j in range(0, len(Q)):
                    H += Q[i][j] * qubo_basis[i] * qubo_basis[j]

            for i in range(0, len(W)):
                H += W[i] * qubo_basis[i]

            # H+= float(bias)

            # if self.verbose: print(H)
            model = H.compile()
            # if self.verbose: print(model)
            bqm = model.to_bqm()
            # if self.verbose: print(bqm)

            # solve the model
            sa = neal.SimulatedAnnealingSampler()

            sampleset = sa.sample(bqm, num_sweeps=100, num_reads=10)

            # if self.verbose: print(sampleset)
            decoded_samples = model.decode_sampleset(sampleset)
            best_sample = min(decoded_samples, key=lambda x: x.energy)

            return best_sample

        def get_qubo_hamiltonian_minimas(self, num_samples):
            """convert factorization machine to qubo hamiltonian and solve the model, getting multiple results"""

            params = copy.deepcopy(self.get_weights())

            V = np.array(params[2], dtype=np.float64)

            W = np.array(np.array(params[1])[0], dtype=np.float64)

            bias = np.array(params[0], dtype=np.float64)[0]

            Q = V @ V.T

            Q = self.VtoQ(V)

            binary_len = int(self.n)
            qubo_basis = []
            for i in range(0, binary_len):
                new_string = "x" + str(i + 1)
                qubo_basis.append(Binary(new_string))

            # define qubo Hamiltonian
            H = 0
            for i in range(0, len(Q)):
                for j in range(0, len(Q)):
                    H += Q[i][j] * qubo_basis[i] * qubo_basis[j]

            for i in range(0, len(W)):
                H += W[i] * qubo_basis[i]

            # H+= float(bias)

            # if self.verbose: print(H)
            model = H.compile()
            # if self.verbose: print(model)
            bqm = model.to_bqm()
            # if self.verbose: print(bqm)

            # solve the model

            sa = neal.SimulatedAnnealingSampler()

            samples_list = []

            for i in range(0, num_samples):
                sampleset = sa.sample(bqm, num_sweeps=1000, num_reads=10)
                # print(sampleset.record['sample'])
                for l in list(sampleset.record["sample"]):
                    samples_list.append(list(l))

            samples_list = [
                list(tupl) for tupl in {tuple(item) for item in samples_list}
            ]

            decoded_samples = model.decode_sampleset(sampleset)
            best_sample = min(decoded_samples, key=lambda x: x.energy)

            return best_sample, decoded_samples

        def predict(self, X):
            self.model.eval()
            with torch.no_grad():
                X = torch.tensor(X, dtype=torch.float32)
                predictions = self.model(X)

                if self.standardscale or self.minmaxscale:
                    predictions = self.scaler.inverse_transform(predictions)

            return predictions
