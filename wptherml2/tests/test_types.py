import numpy as np
import pytest

from wptherml2 import LayerStack, SpectralGrid, TmmRequest


def test_request_validates_shape():
    stack = LayerStack(thickness_m=[0.0, 100e-9, 0.0], materials=["air", "film", "glass"])
    grid = SpectralGrid(wavelength_m=np.array([500e-9, 600e-9]))

    with pytest.raises(ValueError):
        TmmRequest(stack=stack, grid=grid, refractive_index_nk=np.ones((2, 2), dtype=np.complex128))


def test_gradient_layers_default_to_interior_layers():
    stack = LayerStack(thickness_m=[0.0, 100e-9, 200e-9, 0.0])
    grid = SpectralGrid(wavelength_m=np.array([500e-9]))
    refractive_index = np.ones((1, 4), dtype=np.complex128)
    request = TmmRequest(stack=stack, grid=grid, refractive_index_nk=refractive_index)

    np.testing.assert_array_equal(request.gradient_layers, np.array([1, 2]))
