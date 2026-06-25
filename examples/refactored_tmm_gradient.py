"""Thickness-gradient evaluation on the refactored architecture.

Computes the optical spectra together with their derivatives with respect to the
internal-layer thicknesses, using the fast adjoint (prefix/suffix) gradient path.
These are exactly the gradients that drive thickness optimization, but here they
are exposed directly so they can be inspected or fed to a custom optimizer.

The result is also validated against a central finite-difference estimate.
"""

import numpy as np

import wptherml


test_args = {
    "wavelength_list": [400e-9, 1200e-9, 200],
    "material_list": ["Air", "SiO2", "TiO2", "SiO2", "Air"],
    "thickness_list": [0, 230e-9, 120e-9, 230e-9, 0],
    # internal layers whose thickness gradients we want (1-based layer indices)
    "gradient_list": [1, 2, 3],
}

wavelengths = np.linspace(*test_args["wavelength_list"][:2],
                          int(test_args["wavelength_list"][2]))
structure = wptherml.MultilayerStructure.from_spec(
    materials=test_args["material_list"],
    thicknesses=test_args["thickness_list"],
    wavelengths=wavelengths,
)

solver = wptherml.TMMSolver()  # vectorized backend by default
gradient = solver.solve_gradients(
    structure, test_args["gradient_list"], polarizations="p"
)

# dR_dd has shape (number_of_wavelengths, number_of_gradient_layers).
print("dR/dd shape:", gradient.dR_dd.shape)
print("dT/dd shape:", gradient.dT_dd.shape)

# Spot-check against central finite differences for the first gradient layer.
layer = test_args["gradient_list"][0]
step = 1e-12
thicknesses = np.asarray(structure.thicknesses, dtype=float)


def _reflectivity(thicknesses):
    perturbed = wptherml.MultilayerStructure(
        materials=structure.materials,
        thicknesses=thicknesses,
        wavelengths=structure.wavelengths,
        angles=structure.angles,
        refractive_indices=structure.refractive_indices,
    )
    return wptherml.TMMSolver().solve(perturbed, polarizations="p").reflectivity


forward = thicknesses.copy()
backward = thicknesses.copy()
forward[layer] += step
backward[layer] -= step
finite_difference = (_reflectivity(forward) - _reflectivity(backward)) / (2 * step)

analytic = gradient.dR_dd[:, 0]
max_abs_error = np.max(np.abs(analytic - finite_difference))
print(f"max |analytic - finite difference| for layer {layer}: {max_abs_error:.3e}")
