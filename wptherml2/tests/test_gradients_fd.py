import numpy as np

from wptherml2 import LayerStack, SpectralGrid, TmmRequest, compute_gradients


def test_gradient_shapes_match_requested_layers():
    wavelengths = np.linspace(450e-9, 650e-9, 5)
    stack = LayerStack(thickness_m=[0.0, 100e-9, 80e-9, 0.0], materials=["air", "a", "b", "glass"])
    grid = SpectralGrid(wavelength_m=wavelengths, incident_angle_rad=0.1)
    refractive_index = np.column_stack(
        [
            np.ones(wavelengths.size, dtype=np.complex128),
            np.full(wavelengths.size, 2.1 + 0.01j, dtype=np.complex128),
            np.full(wavelengths.size, 1.7 + 0.0j, dtype=np.complex128),
            np.full(wavelengths.size, 1.5 + 0.0j, dtype=np.complex128),
        ]
    )

    result = compute_gradients(
        TmmRequest(
            stack=stack,
            grid=grid,
            refractive_index_nk=refractive_index,
            gradient_layers=np.array([1, 2]),
        )
    )

    assert result.d_reflectivity.shape == (2, wavelengths.size)
    assert result.d_transmissivity.shape == (2, wavelengths.size)
    assert result.d_absorptivity.shape == (2, wavelengths.size)
