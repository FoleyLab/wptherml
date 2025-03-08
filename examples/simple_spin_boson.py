import wptherml
from matplotlib import pyplot as plt
import numpy as np

# dictionaries for case 1
args_1 = {
     "number_of_excitons": 1,
     "number_of_boson_levels": 2,
     "boson_energy_ev": 3.,
     "exciton_energy_ev" : 3.,
     "exciton_boson_coupling_ev" : 0.01
}

sf = wptherml.SpectrumFactory()

# instantiate cases
test_1 = sf.spectrum_factory("Spin-Boson", args_1)



