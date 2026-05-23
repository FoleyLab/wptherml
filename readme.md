WPTherml v1.1.0
===============================

[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/FoleyLab/wptherml/workflows/CI/badge.svg)](https://github.com/FoleyLab/wptherml/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/FoleyLab/wptherml/branch/main/graph/badge.svg)](https://codecov.io/gh/FoleyLab/wptherml/branch/main)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![Open Source Love](https://firstcontributions.github.io/open-source-badges/badges/open-source-v1/open-source.svg)](https://github.com/firstcontributions/open-source-badges)
[![Github Downloads All Releases](https://img.shields.io/github/downloads/FoleyLab/wptherml/total)](https://github.com/FoleyLab/wptherml/releases)
[![Release to PyPI](https://github.com/FoleyLab/wptherml/actions/workflows/release.yml/badge.svg?branch=main)](https://github.com/FoleyLab/wptherml/actions/workflows/release.yml)
[![Documentation Status](https://readthedocs.org/projects/wptherml/badge/?version=latest)](https://wptherml.readthedocs.io/en/latest/?badge=latest)

A Python package for modeling electromagnetic response, thermal radiation, and optimization of multilayer nanophotonic structures.

### Vectorized TMM Example

This example builds a multilayer stack, computes reflectivity, transmissivity,
and emissivity with the refactored vectorized TMM solver, and then evaluates a
selective-mirror figure of merit. During the current migration phase,
`SpectrumFactory` is used only to resolve material names/files into refractive
indices; the optical solve itself is performed by `TMMSolver`.

```python
import contextlib
import io

import numpy as np
import wptherml


# Define the stack and wavelength grid in familiar wptherml units.
materials = ["Air", "SiO2", "Ag", "Air"]
thicknesses = np.array([0, 200e-9, 10e-9, 0])
wavelengths = np.linspace(300e-9, 6000e-9, 1000)

# Temporary material-lookup bridge: this creates the refractive-index array
# used by the new solver. The stdout redirect just hides legacy status prints.
legacy_args = {
    "material_list": materials,
    "thickness_list": thicknesses,
    "wavelength_list": [wavelengths[0], wavelengths[-1], len(wavelengths)],
}
with contextlib.redirect_stdout(io.StringIO()):
    material_context = wptherml.SpectrumFactory().spectrum_factory("Tmm", legacy_args)

# New structure I/O object.
structure = wptherml.MultilayerStructure(
    materials=materials,
    thicknesses=thicknesses,
    wavelengths=wavelengths,
    angles=np.array([0.0]),  # radians
    refractive_indices=material_context._refractive_index_array,
)

# The default TMMSolver backend is "vectorized".
spectrum = wptherml.TMMSolver().solve(structure, polarizations="p")

reflectivity = spectrum.reflectivity
transmissivity = spectrum.transmissivity
emissivity = spectrum.emissivity

# Define simple boxcar windows for the selective mirror objective:
# transmit in the visible, reflect in the 2000-2400 cm^-1 infrared band.
transmissive_envelope = np.where(
    (wavelengths >= 350e-9) & (wavelengths <= 700e-9), 1.0, 0.0
)
reflective_start = 10000000 / 2400 * 1e-9
reflective_stop = 10000000 / 2000 * 1e-9
reflective_envelope = np.where(
    (wavelengths >= reflective_start) & (wavelengths <= reflective_stop), 1.0, 0.0
)

objective = wptherml.SelectiveMirrorObjective(
    transmissive_envelope=transmissive_envelope,
    reflective_envelope=reflective_envelope,
    transmission_efficiency_weight=1 / 3,
    reflection_efficiency_weight=1 / 3,
    reflection_selectivity_weight=1 / 3,
)

selective_mirror_fom = objective.evaluate(spectrum)

print(reflectivity[:3])
print(transmissivity[:3])
print(emissivity[:3])
print(selective_mirror_fom)
```

### [Quickstart](https://github.com/FoleyLab/wptherml/blob/main/docs/quickstart.rst)

### [Examples](https://github.com/FoleyLab/wptherml/tree/main/examples)

### Copyright

Copyright (c) 2022, Foley Lab, The University of North Carolina Charlotte, NC, USA.

#### Acknowledgements
Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.5.
