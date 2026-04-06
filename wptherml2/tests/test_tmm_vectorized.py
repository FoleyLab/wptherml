import numpy as np

from wptherml2 import LayerStack, SpectralGrid, TmmRequest, compute_observables


def test_scalar_and_vectorized_backends_agree():
    wavelengths = np.linspace(450e-9, 750e-9, 9)
    stack = LayerStack(thickness_m=[0.0, 120e-9, 0.0], materials=["air", "film", "glass"])
    grid = SpectralGrid(wavelength_m=wavelengths, incident_angle_rad=0.2, polarization="p")

    refractive_index = np.column_stack(
        [
            np.ones(wavelengths.size, dtype=np.complex128),
            np.full(wavelengths.size, 2.0 + 0.0j, dtype=np.complex128),
            np.full(wavelengths.size, 1.5 + 0.0j, dtype=np.complex128),
        ]
    )

    scalar = compute_observables(TmmRequest(stack=stack, grid=grid, refractive_index_nk=refractive_index, backend="scalar"))
    vectorized = compute_observables(
        TmmRequest(stack=stack, grid=grid, refractive_index_nk=refractive_index, backend="vectorized")
    )

    np.testing.assert_allclose(vectorized.reflectivity, scalar.reflectivity, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(vectorized.transmissivity, scalar.transmissivity, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(vectorized.absorptivity, scalar.absorptivity, rtol=1e-10, atol=1e-10)
