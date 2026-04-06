import numpy as np

from wptherml2 import LayerStack, SpectralGrid, TmmRequest, compute_observables


def test_energy_is_conserved_for_lossless_stack():
    wavelengths = np.linspace(400e-9, 800e-9, 11)
    stack = LayerStack(thickness_m=[0.0, 110e-9, 160e-9, 0.0], materials=["air", "a", "b", "glass"])
    grid = SpectralGrid(wavelength_m=wavelengths, incident_angle_rad=0.3, polarization="s")
    refractive_index = np.column_stack(
        [
            np.ones(wavelengths.size, dtype=np.complex128),
            np.full(wavelengths.size, 2.0 + 0.0j, dtype=np.complex128),
            np.full(wavelengths.size, 1.4 + 0.0j, dtype=np.complex128),
            np.full(wavelengths.size, 1.5 + 0.0j, dtype=np.complex128),
        ]
    )

    result = compute_observables(TmmRequest(stack=stack, grid=grid, refractive_index_nk=refractive_index))

    np.testing.assert_allclose(
        result.reflectivity + result.transmissivity + result.absorptivity,
        1.0,
        atol=1e-10,
    )
