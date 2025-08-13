import time
import numpy as np
import pandas as pd
import math
import argparse
import sys
import matplotlib.pyplot as plt

try:
    import wptherml
except Exception as e:
    print("\n[!] Could not import wptherml. Install it before running: pip install wptherml\n")
    raise

# ------------------------------
# Helpers to build test systems
# ------------------------------

def build_stack(n_layers: int,
                wl_count: int,
                wl_min: float = 400e-9,
                wl_max: float = 800e-9,
                mat_a: str = "SiO2",
                mat_b: str = "TiO2",
                t_a: float = 200e-9,
                t_b: float = 10e-9):
    """Construct test_args for wptherml given number of finite layers and wavelengths.

    n_layers: number of finite layers (no. of thickness entries excluding the two boundary airs)
    wl_count: number of wavelengths to sample between wl_min and wl_max (inclusive)
    The stack alternates mat_a/mat_b starting with mat_a.
    """
    if n_layers <= 0:
        raise ValueError("n_layers must be >= 1")
    if wl_count <= 1:
        raise ValueError("wl_count must be > 1")

    # Wavelength list uses the wptherml shorthand [start, end, N]
    wavelength_list = [wl_min, wl_max, wl_count]

    # Materials: Air at both ends, then alternating A/B for n_layers
    mats = ["Air"]
    thks = [0.0]
    for i in range(n_layers):
        if i % 2 == 0:
            mats.append(mat_a)
            thks.append(t_a)
        else:
            mats.append(mat_b)
            thks.append(t_b)
    mats.append("Air")
    thks.append(0.0)

    test_args = {
        "wavelength_list": wavelength_list,
        "material_list": mats,
        "thickness_list": thks,
    }
    return test_args


def time_once(sf: SpectrumFactory, n_layers: int, wl_count: int, atol=1e-5, rtol=1e-5, check=True):
    """Time one initialization of serial and vectorized codes for the same stack."""
    args = build_stack(n_layers=n_layers, wl_count=wl_count)

    t0 = time.perf_counter()
    test_serial = sf.spectrum_factory('Tmm', args)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    test_vec = sf.spectrum_factory('VecTmm', args)
    t3 = time.perf_counter()

    if check:
        if not np.allclose(test_serial.reflectivity_array, test_vec.reflectivity_array, rtol, atol):
            raise AssertionError("Serial and vectorized reflectivity arrays differ beyond tolerances.")

    return (t1 - t0), (t3 - t2)


def benchmark_grid(sf: SpectrumFactory,
                   wl_counts,
                   layer_counts,
                   repeats: int = 3,
                   warmups: int = 1,
                   check=True,
                   verbose=False):
    """Run timing grid and return a tidy DataFrame."""
    rows = []

    # Warmups on a moderate size to JIT/prime any caches
    for _ in range(max(1, warmups)):
        try:
            time_once(sf, n_layers=max(2, layer_counts[min(1, len(layer_counts)-1)]), wl_count=max(100, wl_counts[min(1, len(wl_counts)-1)]), check=False)
        except Exception:
            pass

    # 1) Scaling with number of wavelengths (layers fixed)
    fixed_layers = layer_counts[-1] if len(layer_counts) > 0 else 18
    for Nw in wl_counts:
        times_serial = []
        times_vec = []
        for r in range(repeats):
            ts, tv = time_once(sf, n_layers=fixed_layers, wl_count=Nw, check=check)
            times_serial.append(ts)
            times_vec.append(tv)
            if verbose:
                print(f"[λ-scale] layers={fixed_layers:2d} Nw={Nw:6d} run={r+1}/{repeats}: serial={ts:.4f}s vec={tv:.4f}s")
        rows.append({"sweep": "wavelengths", "n_layers": fixed_layers, "n_wavelengths": Nw,
                     "mode": "serial", "time_s": float(np.median(times_serial))})
        rows.append({"sweep": "wavelengths", "n_layers": fixed_layers, "n_wavelengths": Nw,
                     "mode": "vectorized", "time_s": float(np.median(times_vec))})

    # 2) Scaling with number of layers (wavelengths fixed)
    fixed_wl = wl_counts[-1] if len(wl_counts) > 0 else 1000
    for L in layer_counts:
        times_serial = []
        times_vec = []
        for r in range(repeats):
            ts, tv = time_once(sf, n_layers=L, wl_count=fixed_wl, check=check)
            times_serial.append(ts)
            times_vec.append(tv)
            if verbose:
                print(f"[L-scale] layers={L:2d} Nw={fixed_wl:6d} run={r+1}/{repeats}: serial={ts:.4f}s vec={tv:.4f}s")
        rows.append({"sweep": "layers", "n_layers": L, "n_wavelengths": fixed_wl,
                     "mode": "serial", "time_s": float(np.median(times_serial))})
        rows.append({"sweep": "layers", "n_layers": L, "n_wavelengths": fixed_wl,
                     "mode": "vectorized", "time_s": float(np.median(times_vec))})

    df = pd.DataFrame(rows)
    return df


def make_default_grids():
    # Reasonable defaults. Adjust if your machine/library can handle more.
    wl_counts = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000]
    layer_counts = [2, 4, 6, 8, 10, 12, 16, 18, 20, 24, 30, 40]
    return wl_counts, layer_counts


def plot_results(df: pd.DataFrame, save_prefix: str | None = None):
    # 1) Scaling with wavelengths
    sub = df[df["sweep"] == "wavelengths"].copy()
    if not sub.empty:
        fig1 = plt.figure()
        for mode in ["serial", "vectorized"]:
            d = sub[sub["mode"] == mode].sort_values("n_wavelengths")
            plt.plot(d["n_wavelengths"].values, d["time_s"].values, marker="o", label=mode)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Number of wavelengths")
        plt.ylabel("Median init time (s)")
        layers_val = int(sub["n_layers"].iloc[0]) if not sub.empty else 0
        plt.title(f"Initialization time vs. wavelengths (layers={layers_val})")
        plt.legend()
        if save_prefix:
            fig1.savefig(f"{save_prefix}_vs_wavelengths.png", dpi=150, bbox_inches="tight")

    # 2) Scaling with layers
    sub = df[df["sweep"] == "layers"].copy()
    if not sub.empty:
        fig2 = plt.figure()
        for mode in ["serial", "vectorized"]:
            d = sub[sub["mode"] == mode].sort_values("n_layers")
            plt.plot(d["n_layers"].values, d["time_s"].values, marker="o", label=mode)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Number of layers (finite)")
        plt.ylabel("Median init time (s)")
        wl_val = int(sub["n_wavelengths"].iloc[0]) if not sub.empty else 0
        plt.title(f"Initialization time vs. layers (Nw={wl_val})")
        plt.legend()
        if save_prefix:
            fig2.savefig(f"{save_prefix}_vs_layers.png", dpi=150, bbox_inches="tight")


def summarize_speedups(df: pd.DataFrame):
    summaries = []

    for sweep in ["wavelengths", "layers"]:
        sub = df[df["sweep"] == sweep]
        if sub.empty:
            continue
        # Compute speedup = serial / vectorized for each matching point
        key = "n_wavelengths" if sweep == "wavelengths" else "n_layers"
        fixed_key = "n_layers" if sweep == "wavelengths" else "n_wavelengths"
        pairs = sub.groupby([key, fixed_key])
        for (k, f), g in pairs:
            try:
                t_serial = g[g["mode"] == "serial"]["time_s"].iloc[0]
                t_vec = g[g["mode"] == "vectorized"]["time_s"].iloc[0]
                speedup = t_serial / t_vec if t_vec > 0 else np.nan
                summaries.append({"sweep": sweep, key: k, fixed_key: f,
                                  "serial_s": t_serial, "vectorized_s": t_vec,
                                  "speedup": speedup})
            except Exception:
                pass

    return pd.DataFrame(summaries)


def main():
    parser = argparse.ArgumentParser(description="Benchmark wptherml Tmm vs VecTmm initialization times.")
    parser.add_argument("--repeats", type=int, default=3, help="Repeat count per grid point; median is reported")
    parser.add_argument("--warmups", type=int, default=1, help="Warmups before timing")
    parser.add_argument("--no-check", action="store_true", help="Disable reflectivity equality check (faster)")
    parser.add_argument("--out", type=str, default="tmm_timing_results.csv", help="CSV output filename")
    parser.add_argument("--save-plots", action="store_true", help="Save PNG plots alongside CSV")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--wl-min", type=float, default=400e-9)
    parser.add_argument("--wl-max", type=float, default=800e-9)
    parser.add_argument("--mat-a", type=str, default="SiO2")
    parser.add_argument("--mat-b", type=str, default="TiO2")
    parser.add_argument("--t-a", type=float, default=200e-9)
    parser.add_argument("--t-b", type=float, default=10e-9)
    parser.add_argument("--wl-counts", type=str, default="50,100,200,500,1000,2000,5000,10000,20000,30000",
                        help="Comma-separated list of wavelength counts")
    parser.add_argument("--layer-counts", type=str, default="2,4,6,8,10,12,16,18,20,24,30,40",
                        help="Comma-separated list of layer counts (finite layers)")

    args = parser.parse_args()

    # Override build_stack defaults if user changed values
    global build_stack
    def build_stack(n_layers: int, wl_count: int,
                    wl_min=args.wl_min, wl_max=args.wl_max,
                    mat_a=args.mat_a, mat_b=args.mat_b,
                    t_a=args.t_a, t_b=args.t_b):
        if n_layers <= 0:
            raise ValueError("n_layers must be >= 1")
        if wl_count <= 1:
            raise ValueError("wl_count must be > 1")
        wavelength_list = [wl_min, wl_max, wl_count]
        mats = ["Air"]
        thks = [0.0]
        for i in range(n_layers):
            if i % 2 == 0:
                mats.append(mat_a)
                thks.append(t_a)
            else:
                mats.append(mat_b)
                thks.append(t_b)
        mats.append("Air")
        thks.append(0.0)
        return {"wavelength_list": wavelength_list, "material_list": mats, "thickness_list": thks}

    wl_counts = [int(x) for x in args.wl_counts.split(',') if x]
    layer_counts = [int(x) for x in args.layer_counts.split(',') if x]

    sf = SpectrumFactory.SpectrumFactory()

    df = benchmark_grid(sf,
                        wl_counts=wl_counts,
                        layer_counts=layer_counts,
                        repeats=args.repeats,
                        warmups=args.warmups,
                        check=not args.no_check,
                        verbose=args.verbose)

    df.to_csv(args.out, index=False)
    print(f"\nSaved results to {args.out}")

    speed = summarize_speedups(df)
    if not speed.empty:
        print("\nSpeedups (serial / vectorized):")
        # Print a few key rows
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None,
                               'display.width', 120):
            print(speed.head(20))
        speed.to_csv(args.out.replace('.csv', '_speedups.csv'), index=False)
        print(f"Saved speedup table to {args.out.replace('.csv', '_speedups.csv')}")

    if args.save_plots:
        prefix = args.out.replace('.csv', '')
        plot_results(df, save_prefix=prefix)
        print("Saved plots.")


if __name__ == "__main__":
    main()

