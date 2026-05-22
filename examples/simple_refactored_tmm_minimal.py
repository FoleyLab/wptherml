"""Minimal migration from legacy TMM arrays to the new spectrum object.

This example is intentionally very close to ``examples/simple_tmm.py``.
The public ``SpectrumFactory`` call is unchanged, so existing scripts can move
incrementally. The difference is that spectra are read from ``ts.spectrum``,
which is the new ``OpticalSpectrum`` result container.
"""

import wptherml
from matplotlib import pyplot as plt


test_args = {
    # Range of wavelengths in meters to compute TMM quantities.
    "wavelength_list": [400e-9, 800e-9, 100],
    # First and last materials should be non-absorbing terminal media.
    "material_list": ["Air", "SiO2", "Au", "Air"],
    # Thicknesses in meters. Terminal layers are semi-infinite, so use 0.
    "thickness_list": [0, 200e-9, 10e-9, 0],
}


# The legacy factory remains supported. Internally, TmmDriver now computes its
# optical result through the refactored TMM solver layer.
sf = wptherml.SpectrumFactory()
ts = sf.spectrum_factory("Tmm", test_args)

# New I/O: use the OpticalSpectrum object instead of reading driver arrays
# directly. The old attributes still exist for compatibility, but this is the
# preferred migration shape for code that still constructs a legacy driver.
spectrum = ts.spectrum

plt.plot(spectrum.wavelengths, spectrum.reflectivity, label="Reflectivity")
plt.plot(spectrum.wavelengths, spectrum.transmissivity, label="Transmissivity")
plt.plot(spectrum.wavelengths, spectrum.emissivity, label="Emissivity")
plt.xlabel("Wavelength (m)")
plt.ylabel("Optical response")
plt.legend()
plt.show()
