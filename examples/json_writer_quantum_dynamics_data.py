import json

# create new dictionary to store calculation data
TC_data = {

    "spin_1_initial_state" : [1, 0],
    "spin_2_initial_state" : [0, 1],
    "cavity_initial_state" : [1, 0],
    "spin_frequency" : 0.5,
    "cavity_frequency" : 0.5,
    "cavity_coupling" : 0.02,
    "cavity_spontaneous_emission" : 0,
    "spin_spontaneous_emission" : 0,
    "cavity_dephasing" : 0,
    "spin_dephasing" : 0,
    "time_steps" : 1000,
    "time_step_size" : 1.0,
    "population_s1g_s2g_cg" : [],
    "population_s1e_s2g_cg" : [],
    "population_s1g_s2e_cg" : [],
    "population_s1e_s2e_cg" : [],
    "population_s1g_s2g_ce" : [],
    "population_s1e_s2g_ce" : [],
    "population_s1g_s2e_ce" : [],
    "population_s1e_s2e_ce" : [],
}

# run dynamics and store the results in the dictionary
TC_data["population_s1g_s2g_cg"] = #< Insert value from appropriate diagonal element of density matrix (spin 1, spin 2, cavity in ground) 
TC_data["population_s1e_s2g_cg"] = #< Insert value from appropriate diagonal element of density matrix (spin 1 excited, spin 2 ground, cavity ground)
TC_data["population_s1g_s2e_cg"] = #< Insert value from appropriate diagonal element of density matrix (spin 1 ground, spin 2 excited, cavity ground)
TC_data["population_s1e_s2e_cg"] = #< Insert value from appropriate diagonal element of density matrix (spin 1 excited, spin 2 excited, cavity ground)
TC_data["population_s1g_s2g_ce"] = #< Insert value from appropriate diagonal element of density matrix (spin 1 ground, spin 2 ground, cavity excited)
TC_data["population_s1e_s2g_ce"] = #< Insert value from appropriate diagonal element of density matrix (spin 1 excited, spin 2 ground, cavity excited)
TC_data["population_s1g_s2e_ce"] = #< Insert value from appropriate diagonal element of density matrix (spin 1 ground, spin 2 excited, cavity excited)
TC_data["population_s1e_s2e_ce"] = #< Insert value from appropriate diagonal element of density matrix (spin 1 excited, spin 2 excited, cavity excited)

# write the data to a JSON file
def write_to_json(data, filename):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def create_output_filename(dictionary):
    # define output file name based on the parameters of the simulation
    output_filename = "TC_simulation_"
    output_filename += "spin_freq_" + str(dictionary["spin_frequency"]) + "_"
    output_filename += "cavity_freq_" + str(dictionary["cavity_frequency"]) + "_"
    output_filename += "cavity_coupling_" + str(dictionary["cavity_coupling"]) + "_"
    output_filename += "cavity_spontaneous_emission_" + str(dictionary["cavity_spontaneous_emission"]) + "_"
    output_filename += "spin_spontaneous_emission_" + str(dictionary["spin_spontaneous_emission"]) + "_"
    output_filename += "cavity_dephasing_" + str(dictionary["cavity_dephasing"]) + "_"
    output_filename += "spin_dephasing_" + str(dictionary["spin_dephasing"]) + ".json"
    return output_filename

output_filename = create_output_filename(TC_data)

write_to_json(TC_data, output_filename)

