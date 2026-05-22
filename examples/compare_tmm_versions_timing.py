"""Compare legacy, serial, and vectorized TMM timing for one multilayer stack.

This example intentionally separates material setup from solver timing:

1. The legacy public driver is constructed once through ``SpectrumFactory``.
   That gives us the refractive-index array that all three calculations use.
2. The timed calls then measure only spectrum calculation work.

The three paths compared are:

* legacy:     ``SpectrumFactory().spectrum_factory("Tmm", args)`` public driver API
* serial:     ``TMMSolver(backend="serial")`` on a ``MultilayerStructure``
* vectorized: ``TMMSolver(backend="vectorized")`` on the same structure

Run from the repository root with:

    python examples/compare_tmm_versions_timing.py
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


NUMBER_OF_INTERNAL_LAYERS = 10
SIO2_THICKNESS = 100e-9
AG_THICKNESS = 10e-9
WAVELENGTH_LIST = [400e-9, 2_000e-9, 1_000]
REPEATS = 10


def build_stack() -> tuple[list[str], list[float]]:
    """Build Air / (SiO2 / Ag)^5 / Air with matching thicknesses.

    The user-facing stack has ten finite internal layers. The terminal Air
    layers are included because the TMM drivers expect semi-infinite incident
    and exit media; their thicknesses are therefore zero.
    """

    internal_materials = [
        "SiO2" if layer_index % 2 == 0 else "Ag"
        for layer_index in range(NUMBER_OF_INTERNAL_LAYERS)
    ]
    internal_thicknesses = [
        SIO2_THICKNESS if material == "SiO2" else AG_THICKNESS
        for material in internal_materials
    ]

    materials = ["Air", *internal_materials, "Air"]
    thicknesses = [0.0, *internal_thicknesses, 0.0]
    return materials, thicknesses


def build_driver_args() -> dict[str, object]:
    """Return the dictionary accepted by the legacy public driver API."""

    materials, thicknesses = build_stack()
    return {
        "wavelength_list": WAVELENGTH_LIST,
        "material_list": materials,
        "thickness_list": thicknesses,
        # Keep the comparison simple: normal incidence, p polarization.
        "incident_angle": 0.0,
        "polarization": "p",
    }


def build_structure_from_driver(driver) -> MultilayerStructure:
    """Reuse the legacy driver's material lookup for direct solver examples."""

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
    args = build_driver_args()

    # The legacy constructor is intentionally outside the timed region. It does
    # material interpolation, object setup, and one initial spectrum calculation.
    # We silence its status prints so the timing table stays readable.
    with contextlib.redirect_stdout(io.StringIO()):
        legacy_driver = wptherml.SpectrumFactory().spectrum_factory("Tmm", args)

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

    print("TMM timing comparison")
    print("=====================")
    print(f"Internal layers: {NUMBER_OF_INTERNAL_LAYERS}")
    print(f"Stack: {' / '.join(args['material_list'])}")
    print(
        "Thicknesses: "
        + ", ".join(f"{thickness * 1e9:g} nm" for thickness in args["thickness_list"])
    )
    print(
        "Wavelength grid: "
        f"{WAVELENGTH_LIST[0] * 1e9:g}-{WAVELENGTH_LIST[1] * 1e9:g} nm, "
        f"{WAVELENGTH_LIST[2]} points"
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
