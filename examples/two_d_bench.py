import wptherml
from matplotlib import pyplot as plt
import numpy as np
import time
from matplotlib.ticker import ScalarFormatter, LogLocator
from matplotlib.colors import LogNorm
import numpy.ma as ma


# Example input dictionary
test_args = {
    "wavelength_list": [400e-9, 800e-9, 1000],  # 100 wavelengths from 400 to 800 nm
    "material_list": ["Air", "SiO2", "TiO2", "Air"],
    "thickness_list": [0, 200e-9, 10e-9, 0],
}

sf = wptherml.SpectrumFactory()


def build_stack(n_layers, mat_a="SiO2", mat_b="TiO2", ri_a = 1.5, ri_b = 2.4, t_a=200e-9, t_b=10e-9):
    """
    Build material_list and thickness_list for n_layers finite layers.
    Alternates mat_a and mat_b, with Air on both sides.
    """
    material_list = ["Air"]
    thickness_list = [0.0]

    for i in range(n_layers):
        if i % 2 == 0:
            material_list.append(mat_a)
            thickness_list.append(t_a)
        else:
            material_list.append(mat_b)
            thickness_list.append(t_b)

    material_list.append("Air")
    thickness_list.append(0.0)

    return material_list, thickness_list



# Collect all measurements in a tidy list of dicts
records = []
for n_layers in [4, 10, 12, 16, 18, 20, 22, 24, 26, 28, 30, 35, 40, 45, 50, 55, 60, 65, 70]:
    mats, thks = build_stack(n_layers)
    for nwl in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
        test_args = {
            "wavelength_list": [400e-9, 800e-9, nwl],
            "material_list": mats,
            "thickness_list": thks
        }

        start = time.time()
        test_serial = sf.spectrum_factory('Tmm', test_args)
        serial_init = time.time() - start

        start = time.time()
        test_vec = sf.spectrum_factory('VecTmm', test_args)
        vec_init = time.time() - start

        records.append({
            "layers": n_layers,
            "wavelengths": nwl,
            "serial_s": serial_init,
            "vectorized_s": vec_init,
            "speedup": vec_init / max(serial_init, 1e-15)  # guard against 0
        })

layers = np.unique([r["layers"] for r in records])
wls    = np.unique([r["wavelengths"] for r in records])

# map from (row=wavelength index, col=layer index) -> speedup
speedup_grid = np.full((len(wls), len(layers)), np.nan, dtype=float)

for r in records:
    i = np.where(wls    == r["wavelengths"])[0][0]  # row index (wavelengths)
    j = np.where(layers == r["layers"])[0][0]       # col index (layers)
    speedup_grid[i, j] = r["speedup"]
    


def plot_speedup_matrix(speedup_grid, wls, layers,
                        title="Speedup (Serial / Vectorized) vs. Wavelengths × Layers",
                        x_label="Number of finite layers",
                        y_label="Number of wavelengths",
                        log_colors=True,
                        annotate=False,
                        save_prefix=None):
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
    })

    # Mask NaNs so they show as a neutral color
    data = ma.masked_invalid(speedup_grid)

    # Choose color normalization
    if log_colors:
        # Use log color scale if speedups span decades
        finite_vals = data.compressed()
        vmin = np.percentile(finite_vals, 5) if finite_vals.size else 1.0
        vmax = np.percentile(finite_vals, 95) if finite_vals.size else 10.0
        vmin = max(vmin, 1e-3)
        norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin*1.01))
    else:
        norm = None

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        norm=norm,
        cmap="viridis"
    )

    # Axis ticks show actual values (rows=wavelengths, cols=layers)
    ax.set_xticks(np.arange(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(wls)))
    ax.set_yticklabels(wls)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Speedup (serial / vectorized)")

    # Optional per-cell annotation
    if annotate:
        for i in range(speedup_grid.shape[0]):
            for j in range(speedup_grid.shape[1]):
                val = speedup_grid[i, j]
                if np.isfinite(val):
                    # Lightly formatted, fewer decimals for readability
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8, color="white")

    plt.tight_layout()
    if save_prefix:
        fig.savefig(f"{save_prefix}.png")
        fig.savefig(f"{save_prefix}.svg")
    plt.show()

# Usage
plot_speedup_matrix(speedup_grid, wls, layers, log_colors=True, annotate=False, save_prefix="speedup_heatmap_wl_layers")