"""The optimizer should use the vectorized solver and reduce the objective."""

import contextlib
import io

import numpy as np

import wptherml


def _opt_args(extra=None):
    args = {
        "wavelength_list": [300e-9, 6000e-9, 200],
        "material_list": ["Air", "SiO2", "TiO2", "SiO2", "Air"],
        "thickness_list": [0, 300e-9, 200e-9, 300e-9, 0],
        "reflective_window_wn": [2000, 2400],
        "transmissive_window_nm": [350, 700],
        "gradient_list": [1, 2, 3],
        "lower_bound": 10,
        "upper_bound": 1000,
    }
    if extra:
        args.update(extra)
    return args


def _make_optdriver(args):
    with contextlib.redirect_stdout(io.StringIO()):
        return wptherml.SpectrumFactory().spectrum_factory("Opt", args)


def test_optimizer_defaults_to_vectorized_backend():
    opt = _make_optdriver(_opt_args())
    assert opt.optimization_backend == "vectorized"
    assert opt.solver.backend == "vectorized"


def test_optimization_backend_is_configurable():
    opt = _make_optdriver(_opt_args({"optimization_backend": "serial"}))
    assert opt.solver.backend == "serial"


def test_vectorized_and_serial_optimizer_agree_on_fom_and_gradient():
    vec = _make_optdriver(_opt_args())
    ser = _make_optdriver(_opt_args({"optimization_backend": "serial"}))

    x0 = vec.thickness_array[1:-1] * 1e9
    with contextlib.redirect_stdout(io.StringIO()):
        fom_v, grad_v = vec.compute_fom_and_gradient_from_thickness_array(x0)
        fom_s, grad_s = ser.compute_fom_and_gradient_from_thickness_array(x0)

    assert np.isclose(fom_v, fom_s, rtol=1e-8)
    assert np.allclose(grad_v, grad_s, rtol=1e-5, atol=1e-8)


def test_bfgs_optimization_improves_objective():
    opt = _make_optdriver(_opt_args())
    x0 = opt.thickness_array[1:-1] * 1e9
    with contextlib.redirect_stdout(io.StringIO()):
        fom_start, _ = opt.compute_fom_and_gradient_from_thickness_array(x0)
        opt.optimize_bfgs()
        # optimize_bfgs leaves the driver at the final thicknesses
        x_final = opt.thickness_array[1:-1] * 1e9
        fom_final, _ = opt.compute_fom_and_gradient_from_thickness_array(x_final)

    # Maximization is encoded as minimizing -FOM, so the stored fom should not
    # increase (objective improved or held).
    assert fom_final <= fom_start + 1e-9
