import wpspecdev
from matplotlib import pyplot as plt
test_1_args = {
    "wavelength_list": [400e-9, 800e-9, 1000],
    "material_list": [
        "Air",
        "Au",
        "Air",
    ],
    "thickness_list": [0, 200e-9, 0],
}

sf = wpspecdev.SpectrumFactory()  
test = sf.spectrum_factory('Tmm', test_1_args)
test.render_color("Tritanopia", colorblindness="Tritanopia")
test.render_color("Deuteranopia", colorblindness="Deuteranopia")
test.render_color("Protanopia", colorblindness="Protanopia")
test.render_color("Full Color Vision", colorblindness="False")

 # Protanopia Deuteranopia Tritanopia
#print(test.reflectivity_array[0])
#mt_5.compute_hamiltonian( )



