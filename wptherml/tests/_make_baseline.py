"""Capture a frozen numerical snapshot of the TMM kernels from the current code.

Run with the *current* (known-good) implementation to produce
``_baseline_reference.npz``. The ``test_tmm_core_numerics`` regression test then
asserts that future kernel rewrites reproduce these values to tight tolerance.

Usage:
    python -m wptherml.tests._make_baseline
"""

import os

import numpy as np

import wptherml


REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "_baseline_reference.npz")


def _reference_args():
    return {
        # Mid-size grid that exercises the wavelength-batched path.
        "wavelength_list": [300e-9, 6000e-9, 200],
        # A few absorbing + dielectric layers, several gradient layers.
        "material_list": ["Air", "SiO2", "TiO2", "Ag", "SiO2", "Air"],
        "thickness_list": [0, 230e-9, 120e-9, 8e-9, 180e-9, 0],
        "gradient_list": [1, 2, 3, 4],
    }


def build_reference():
    args = _reference_args()
    driver = wptherml.SpectrumFactory().spectrum_factory("Tmm", args)

    structure = wptherml.MultilayerStructure(
        materials=driver.material_array,
        thicknesses=driver.thickness_array,
        wavelengths=driver.wavelength_array,
        angles=np.array([0.0, np.deg2rad(35.0), np.deg2rad(60.0)]),
        refractive_indices=driver._refractive_index_array,
    )

    data = {
        "wavelengths": structure.wavelengths,
        "angles": structure.angles,
        "thicknesses": structure.thicknesses,
        "refractive_indices": structure.refractive_indices,
        "gradient_layers": np.array(args["gradient_list"]),
    }

    # Spectra for both polarizations at multiple angles (no axis squeezing).
    for backend in ("vectorized", "serial"):
        spectrum = wptherml.TMMSolver(backend=backend).solve(
            structure, polarizations=["s", "p"]
        )
        data[f"R_{backend}"] = spectrum.reflectivity
        data[f"T_{backend}"] = spectrum.transmissivity
        data[f"E_{backend}"] = spectrum.emissivity

    # Gradients (vectorized path) for both polarizations.
    gradient = wptherml.TMMSolver(backend="vectorized").solve_gradients(
        structure, args["gradient_list"], polarizations=["s", "p"]
    )
    data["dR_dd"] = gradient.dR_dd
    data["dT_dd"] = gradient.dT_dd
    data["dE_dd"] = gradient.dE_dd

    return data


def main():
    data = build_reference()
    np.savez_compressed(REFERENCE_PATH, **data)
    print(f"wrote {REFERENCE_PATH}")
    for key, value in data.items():
        print(f"  {key}: shape={np.shape(value)}")


if __name__ == "__main__":
    main()
