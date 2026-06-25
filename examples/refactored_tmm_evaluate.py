"""Direct R / T / emissivity evaluation on the refactored architecture.

This mirrors the input -> output convention of the legacy ``simple_tmm.py``
example (a familiar ``test_args`` dictionary), but uses the refactored
``MultilayerStructure -> TMMSolver -> OpticalSpectrum`` workflow. Material names
are resolved directly by ``MultilayerStructure.from_spec`` -- there is no longer
any need to bootstrap refractive indices through ``SpectrumFactory``.
"""

import numpy as np

import wptherml


# Same familiar inputs as the legacy factory examples.
test_args = {
    # range of wavelengths in meters: [start, stop, number_of_points]
    "wavelength_list": [400e-9, 800e-9, 100],
    # first and last materials should be non-absorbing terminal media ("Air")
    "material_list": ["Air", "SiO2", "Au", "Air"],
    # thickness of each layer in meters; terminal layers are semi-infinite (0)
    "thickness_list": [0, 200e-9, 10e-9, 0],
}

# Build the wavelength grid and the structure (refractive indices resolved here).
wavelengths = np.linspace(*test_args["wavelength_list"][:2],
                          int(test_args["wavelength_list"][2]))
structure = wptherml.MultilayerStructure.from_spec(
    materials=test_args["material_list"],
    thicknesses=test_args["thickness_list"],
    wavelengths=wavelengths,
)

# The vectorized backend is the recommended/default path.
spectrum = wptherml.TMMSolver().solve(structure, polarizations="p")

# Same outputs as the legacy driver's *_array attributes, now on a result object.
print("wavelengths :", spectrum.wavelengths[:3], "...")
print("reflectivity:", spectrum.reflectivity[:3], "...")
print("transmissiv.:", spectrum.transmissivity[:3], "...")
print("emissivity  :", spectrum.emissivity[:3], "...")

if __name__ == "__main__":
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        raise SystemExit(0)

    plt.plot(spectrum.wavelengths * 1e9, spectrum.reflectivity, label="Reflectivity")
    plt.plot(spectrum.wavelengths * 1e9, spectrum.transmissivity, label="Transmissivity")
    plt.plot(spectrum.wavelengths * 1e9, spectrum.emissivity, label="Emissivity")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Optical response")
    plt.legend()
    plt.show()
