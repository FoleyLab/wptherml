import wpspecdev
from matplotlib import pyplot as plt
test_1_args = {
    "wavelength_list": [501e-9, 501e-9, 1],
    "material_list": [
        "Air",
        "TiO2",
        "SiO2",
        "Ag",
        "Au",
        "Pt",
        "AlN",
        "Al2O3",
        "Air",
    ],
    "thickness_list": [0, 200e-9, 100e-9, 5e-9, 6e-9, 7e-9, 100e-9, 201e-9, 0],
}

sf = wpspecdev.SpectrumFactory()  
test = sf.spectrum_factory('Tmm', test_1_args)

print(test.reflectivity_array[0])
#mt_5.compute_hamiltonian( )



