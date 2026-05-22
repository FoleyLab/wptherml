"""Use the refactored TMM structure -> solver -> spectrum workflow.

This example keeps the familiar ``test_args`` dictionary from the legacy API,
then converts it into the new I/O objects:

    MultilayerStructure -> TMMSolver -> OpticalSpectrum

During this migration phase, material interpolation still lives on the legacy
driver classes. The small helper below uses ``SpectrumFactory`` only to resolve
material names/files into a refractive-index array. The actual optical solve is
performed by ``TMMSolver`` and returns an ``OpticalSpectrum``.
"""

import contextlib
import io

import numpy as np
import wptherml
from matplotlib import pyplot as plt


test_args = {
    # Range of wavelengths in meters to compute TMM quantities.
    "wavelength_list": [400e-9, 800e-9, 100],
    # First and last materials should be non-absorbing terminal media.
    "material_list": ["Air", "SiO2", "Au", "Air"],
    # Thicknesses in meters. Terminal layers are semi-infinite, so use 0.
    "thickness_list": [0, 200e-9, 10e-9, 0],
    # The new solver expects radians through MultilayerStructure. Keeping the
    # legacy input in degrees here lets old dictionaries remain familiar.
    "incident_angle": 0.0,
    "polarization": "p",
}


def multilayer_from_legacy_args(args):
    """Build a MultilayerStructure from a legacy-style argument dictionary."""

    # The legacy driver prints status messages on construction. Redirecting
    # stdout keeps this example focused on the new solver result.
    with contextlib.redirect_stdout(io.StringIO()):
        material_context = wptherml.SpectrumFactory().spectrum_factory("Tmm", args)

    structure = wptherml.MultilayerStructure(
        materials=material_context.material_array,
        thicknesses=material_context.thickness_array,
        wavelengths=material_context.wavelength_array,
        angles=np.array([material_context.incident_angle]),
        refractive_indices=material_context._refractive_index_array,
    )
    return structure, material_context.polarization


structure, polarization = multilayer_from_legacy_args(test_args)

# The vectorized backend is the recommended/default path.
solver = wptherml.TMMSolver()
spectrum = solver.solve(structure, polarizations=polarization)

# To force the serial backend instead, use:
# serial_spectrum = wptherml.TMMSolver(backend="serial").solve(
#     structure, polarizations=polarization
# )

plt.plot(spectrum.wavelengths, spectrum.reflectivity, label="Reflectivity")
plt.plot(spectrum.wavelengths, spectrum.transmissivity, label="Transmissivity")
plt.plot(spectrum.wavelengths, spectrum.emissivity, label="Emissivity")
plt.xlabel("Wavelength (m)")
plt.ylabel("Optical response")
plt.legend()
plt.show()
