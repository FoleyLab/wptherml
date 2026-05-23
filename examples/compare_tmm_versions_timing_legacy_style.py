"""Legacy-style timing comparison for the refactored TMM paths.

This is a sibling to ``compare_tmm_versions_timing.py`` that keeps the stack
and wavelength specification as close as possible to the classic wptherml
examples:

    test_args = {
        "wavelength_list": [...],
        "material_list": [...],
        "thickness_list": [...],
    }

The structure here is still the requested ten finite-layer alternating
SiO2/Ag stack, but it is written out directly in the legacy dictionary style.
The wavelength grid follows the classic simple TMM example: 400-800 nm with
100 wavelength points.

Run from the repository root with:

    python examples/compare_tmm_versions_timing_legacy_style.py
"""

from __future__ import annotations

import contextlib
import io
import statistics
import sys
import time
from pathlib import Path

import numpy as np

# Make the example work when run directly from a source checkout, without
# requiring an editable install first.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import wptherml
from wptherml.solvers import TMMSolver
from wptherml.structures import MultilayerStructure


REPEATS = 25


test_args = {
    # Legacy-style wavelength range: start, stop, number of points.
    "wavelength_list": [400e-9, 800e-9, 3000],
    # Ten finite internal layers, alternating SiO2 and Ag, with Air terminals.
    "material_list": [
        "Air",
        "SiO2",
        "Ag",
        "SiO2",
        "Ag",
        "SiO2",
        "Ag",
        "SiO2",
        "Ag",
        "SiO2",
        "Ag",
        "Air",
    ],
    # Terminal Air layers are semi-infinite, so their thicknesses are zero.
    # SiO2 layers are 100 nm; Ag layers are 10 nm.
    "thickness_list": [
        0,
        100e-9,
        10e-9,
        100e-9,
        10e-9,
        100e-9,
        10e-9,
        100e-9,
        10e-9,
        100e-9,
        10e-9,
        0,
    ],
    # Keep the comparison simple and identical across all three calculations.
    "incident_angle": 0.0,
    "polarization": "p",
}


def build_structure_from_driver(driver) -> MultilayerStructure:
    """Convert a legacy driver object into the new structure I/O object."""

    return MultilayerStructure(
        materials=driver.material_array,
        thicknesses=driver.thickness_array,
        wavelengths=driver.wavelength_array,
        angles=np.array([driver.incident_angle]),
        refractive_indices=driver._refractive_index_array,
    )


def time_calculation(label: str, calculation, repeats: int = REPEATS) -> dict[str, float]:
    """Run ``calculation`` repeatedly and return timing statistics in seconds."""

    timings = []

    # One warm-up call keeps first-use overhead out of the reported timings.
    calculation()

    for _ in range(repeats):
        start = time.perf_counter()
        calculation()
        timings.append(time.perf_counter() - start)

    return {
        "label": label,
        "best": min(timings),
        "mean": statistics.fmean(timings),
        "stdev": statistics.stdev(timings) if len(timings) > 1 else 0.0,
    }


def max_abs_difference(reference, candidate) -> float:
    """Return the largest absolute spectral difference across R, T, and E."""

    differences = [
        np.max(np.abs(reference.reflectivity - candidate.reflectivity)),
        np.max(np.abs(reference.transmissivity - candidate.transmissivity)),
        np.max(np.abs(reference.emissivity - candidate.emissivity)),
    ]
    return float(max(differences))


def main() -> None:
    # Construct the legacy driver once. This resolves material names into
    # refractive indices and performs one initial spectrum calculation. The
    # timing below measures repeated spectrum solves, not material setup.
    with contextlib.redirect_stdout(io.StringIO()):
        legacy_driver = wptherml.SpectrumFactory().spectrum_factory("Tmm", test_args)

    structure = build_structure_from_driver(legacy_driver)
    serial_solver = TMMSolver(backend="serial")
    vectorized_solver = TMMSolver(backend="vectorized")

    legacy_result = legacy_driver.spectrum
    serial_result = serial_solver.solve(structure, polarizations="p")
    vectorized_result = vectorized_solver.solve(structure, polarizations="p")

    results = [
        time_calculation("legacy public TmmDriver", legacy_driver.compute_spectrum),
        time_calculation(
            "TMMSolver serial backend",
            lambda: serial_solver.solve(structure, polarizations="p"),
        ),
        time_calculation(
            "TMMSolver vectorized backend",
            lambda: vectorized_solver.solve(structure, polarizations="p"),
        ),
    ]

    print("Legacy-style TMM timing comparison")
    print("==================================")
    print(f"Stack: {' / '.join(test_args['material_list'])}")
    print(
        "Thicknesses: "
        + ", ".join(
            f"{thickness * 1e9:g} nm" for thickness in test_args["thickness_list"]
        )
    )
    print(
        "Wavelength grid: "
        f"{test_args['wavelength_list'][0] * 1e9:g}-"
        f"{test_args['wavelength_list'][1] * 1e9:g} nm, "
        f"{test_args['wavelength_list'][2]} points"
    )
    print(f"Timing repeats: {REPEATS}")
    print()
    print(f"{'Calculation':<30} {'best (ms)':>12} {'mean (ms)':>12} {'stdev (ms)':>12}")
    print("-" * 70)

    for row in results:
        print(
            f"{row['label']:<30} "
            f"{row['best'] * 1e3:>12.3f} "
            f"{row['mean'] * 1e3:>12.3f} "
            f"{row['stdev'] * 1e3:>12.3f}"
        )

    print()
    print("Agreement with legacy public driver")
    print("-----------------------------------")
    print(f"serial max |delta|:     {max_abs_difference(legacy_result, serial_result):.3e}")
    print(
        "vectorized max |delta|: "
        f"{max_abs_difference(legacy_result, vectorized_result):.3e}"
    )


if __name__ == "__main__":
    main()
