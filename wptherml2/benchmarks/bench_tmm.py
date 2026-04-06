"""Small benchmark harness for wptherml2."""

from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wptherml2 import LayerStack, SpectralGrid, TmmRequest, compute_observables


def main() -> None:
    wavelengths = np.linspace(400e-9, 900e-9, 2000)
    stack = LayerStack(thickness_m=[0.0, 100e-9, 150e-9, 80e-9, 0.0])
    grid = SpectralGrid(wavelength_m=wavelengths, incident_angle_rad=0.2, polarization="p")
    refractive_index = np.column_stack(
        [
            np.ones(wavelengths.size, dtype=np.complex128),
            np.full(wavelengths.size, 2.0 + 0.02j, dtype=np.complex128),
            np.full(wavelengths.size, 1.45 + 0.0j, dtype=np.complex128),
            np.full(wavelengths.size, 2.2 + 0.01j, dtype=np.complex128),
            np.full(wavelengths.size, 1.5 + 0.0j, dtype=np.complex128),
        ]
    )
    request = TmmRequest(stack=stack, grid=grid, refractive_index_nk=refractive_index)

    start = time.perf_counter()
    result = compute_observables(request)
    elapsed_s = time.perf_counter() - start

    print(f"backend={result.state.backend} wavelengths={wavelengths.size} elapsed_s={elapsed_s:.6f}")


if __name__ == "__main__":
    main()
