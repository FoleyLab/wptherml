import numpy as np

from wptherml2 import LayerStack, SpectralGrid, TmmRequest, compute_observables


def test_observables_are_bounded_for_passive_stack():
    wavelengths = np.linspace(500e-9, 900e-9, 7)
    stack = LayerStack(thickness_m=[0.0, 90e-9, 0.0], materials=["air", "film", "glass"])
    grid = SpectralGrid(wavelength_m=wavelengths)
    refractive_index = np.column_stack(
        [
            np.ones(wavelengths.size, dtype=np.complex128),
            np.full(wavelengths.size, 1.8 + 0.0j, dtype=np.complex128),
            np.full(wavelengths.size, 1.5 + 0.0j, dtype=np.complex128),
        ]
    )

    result = compute_observables(TmmRequest(stack=stack, grid=grid, refractive_index_nk=refractive_index))

    assert np.all(result.reflectivity >= -1e-12)
    assert np.all(result.transmissivity >= -1e-12)
    assert np.all(result.absorptivity >= -1e-12)
    assert np.all(result.reflectivity <= 1.0 + 1e-12)
