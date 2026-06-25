"""Propagate layer-thickness uncertainty through the optical response.

A fabricated stack ``Air | d1 SiO2 | d2 TiO2 | d3 SiO2 | Air`` has manufacturing
tolerances on d1, d2, d3. This example builds the nominal (target) structure,
then uses ``ThicknessEnsemble`` to draw a swarm of replicas whose internal
thicknesses are perturbed about the target by a normal distribution, and solves
the entire swarm in a single vectorized call. It reports the mean spectrum and a
1-sigma band.
"""

import numpy as np

import wptherml


test_args = {
    "wavelength_list": [400e-9, 800e-9, 200],
    "material_list": ["Air", "SiO2", "TiO2", "SiO2", "Air"],
    "thickness_list": [0, 230e-9, 120e-9, 230e-9, 0],
}

wavelengths = np.linspace(*test_args["wavelength_list"][:2],
                          int(test_args["wavelength_list"][2]))
target = wptherml.MultilayerStructure.from_spec(
    materials=test_args["material_list"],
    thicknesses=test_args["thickness_list"],
    wavelengths=wavelengths,
)

# 5 nm (1-sigma) thickness tolerance on every internal layer, 1000 replicas.
ensemble = wptherml.ThicknessEnsemble(
    target,
    sigma=5e-9,
    number_of_samples=1000,
    distribution="normal",
    seed=0,
)
result = ensemble.solve(polarizations="p")

print("reflectivity ensemble shape:", result.reflectivity.shape)  # (1000, 200)
print("mean reflectivity (first 3):", result.reflectivity_mean[:3])
print("std  reflectivity (first 3):", result.reflectivity_std[:3])

if __name__ == "__main__":
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        raise SystemExit(0)

    wl_nm = result.wavelengths * 1e9
    mean = result.reflectivity_mean
    std = result.reflectivity_std
    plt.plot(wl_nm, mean, color="C0", label="mean reflectivity")
    plt.fill_between(wl_nm, mean - std, mean + std, color="C0", alpha=0.3,
                     label=r"$\pm 1\sigma$")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectivity")
    plt.legend()
    plt.show()
