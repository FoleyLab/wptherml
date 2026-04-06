import numpy as np

from wptherml2 import LayerStack, SpectralGrid, TmmRequest, compute_observables


def test_uniform_medium_has_zero_reflection():
    wavelengths = np.linspace(400e-9, 700e-9, 5)
    stack = LayerStack(thickness_m=[0.0, 200e-9, 0.0], materials=["air", "air", "air"])
    grid = SpectralGrid(wavelength_m=wavelengths, incident_angle_rad=0.0, polarization="s")
    refractive_index = np.ones((wavelengths.size, 3), dtype=np.complex128)
    request = TmmRequest(stack=stack, grid=grid, refractive_index_nk=refractive_index, backend="scalar")

    result = compute_observables(request)

    np.testing.assert_allclose(result.reflectivity, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.transmissivity, 1.0, atol=1e-12)
    np.testing.assert_allclose(result.absorptivity, 0.0, atol=1e-12)
