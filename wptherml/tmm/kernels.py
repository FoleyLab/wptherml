"""Core transfer-matrix kernels for scalar and batched TMM workflows.

The routines in this module are intentionally side-effect free and operate on
NumPy arrays only.  Driver classes can use these kernels to build higher-level
APIs while sharing one numerical implementation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.float64]
ArrayC = NDArray[np.complex128]


def compute_k0(wavelength_m: ArrayF) -> ArrayF:
    """Compute free-space wavevector magnitudes.

    Parameters
    ----------
    wavelength_m : numpy.ndarray
        One-dimensional wavelength grid in meters.

    Returns
    -------
    numpy.ndarray
        One-dimensional array of free-space wavevector magnitudes in inverse
        meters with the same shape as ``wavelength_m``.
    """

    return 2.0 * np.pi / wavelength_m


def compute_kx(refractive_index_incident: ArrayC, k0: ArrayF, incident_angle_rad: float) -> ArrayC:
    """Compute in-plane wavevector components.

    Parameters
    ----------
    refractive_index_incident : numpy.ndarray
        Complex refractive index in the incident medium for each wavelength.
    k0 : numpy.ndarray
        Free-space wavevector magnitudes from :func:`compute_k0`.
    incident_angle_rad : float
        Incident angle in radians.

    Returns
    -------
    numpy.ndarray
        Complex in-plane wavevector for each wavelength.
    """

    return refractive_index_incident * np.sin(incident_angle_rad) * k0


def compute_kz(refractive_index: ArrayC, k0: ArrayF, kx: ArrayC) -> ArrayC:
    """Compute normal wavevector components in each layer.

    Parameters
    ----------
    refractive_index : numpy.ndarray
        Complex refractive index array with shape
        ``(number_of_wavelengths, number_of_layers)``.
    k0 : numpy.ndarray
        Free-space wavevector magnitudes.
    kx : numpy.ndarray
        In-plane wavevector components for each wavelength.

    Returns
    -------
    numpy.ndarray
        Complex normal wavevector array with shape
        ``(number_of_wavelengths, number_of_layers)``.
    """

    return np.sqrt((refractive_index * k0[:, np.newaxis]) ** 2 - kx[:, np.newaxis] ** 2)


def compute_dm_scalar(refractive_index: complex, cosine_theta: complex, polarization: str) -> Tuple[ArrayC, ArrayC]:
    """Compute scalar D and D-inverse matrices for one layer.

    Parameters
    ----------
    refractive_index : complex
        Refractive index of the current layer.
    cosine_theta : complex
        Cosine of propagation angle in the current layer.
    polarization : str
        Polarization label (``"s"`` or ``"p"``).

    Returns
    -------
    tuple of numpy.ndarray
        Pair ``(D, D_inv)`` where each matrix has shape ``(2, 2)``.
    """

    dm = np.zeros((2, 2), dtype=np.complex128)
    dim = np.zeros((2, 2), dtype=np.complex128)

    if polarization == "s":
        dm[0, 0] = 1.0
        dm[0, 1] = 1.0
        dm[1, 0] = refractive_index * cosine_theta
        dm[1, 1] = -refractive_index * cosine_theta
    else:
        dm[0, 0] = cosine_theta
        dm[0, 1] = cosine_theta
        dm[1, 0] = refractive_index
        dm[1, 1] = -refractive_index

    det = 1.0 / (dm[0, 0] * dm[1, 1] - dm[0, 1] * dm[1, 0])
    dim[0, 0] = det * dm[1, 1]
    dim[0, 1] = -det * dm[0, 1]
    dim[1, 0] = -det * dm[1, 0]
    dim[1, 1] = det * dm[0, 0]
    return dm, dim


def compute_pm_scalar(phi: complex) -> ArrayC:
    """Compute scalar propagation matrix for one layer.

    Parameters
    ----------
    phi : complex
        Phase accumulation term ``kz * d`` for one layer.

    Returns
    -------
    numpy.ndarray
        Complex propagation matrix with shape ``(2, 2)``.
    """

    pm = np.eye(2, dtype=np.complex128)
    pm[0, 0] = np.exp(-1j * phi)
    pm[1, 1] = np.exp(1j * phi)
    return pm


def compute_pm_batch(phi: ArrayC) -> ArrayC:
    """Compute batched propagation matrices for one layer across wavelengths.

    Parameters
    ----------
    phi : numpy.ndarray
        Phase accumulation term for one layer at each wavelength with shape
        ``(number_of_wavelengths,)``.

    Returns
    -------
    numpy.ndarray
        Batched propagation matrices with shape
        ``(number_of_wavelengths, 2, 2)``.
    """

    pm = np.zeros((phi.shape[0], 2, 2), dtype=np.complex128)
    pm[:, 0, 0] = np.exp(-1j * phi)
    pm[:, 1, 1] = np.exp(1j * phi)
    return pm


def _compute_cos_theta(refractive_index: ArrayC, k0: ArrayF, kz: ArrayC, incident_angle_rad: float) -> ArrayC:
    """Compute cosine of propagation angle in each layer.

    Parameters
    ----------
    refractive_index : numpy.ndarray
        Complex refractive-index array with shape ``(n_wavelengths, n_layers)``.
    k0 : numpy.ndarray
        Free-space wavevector magnitudes.
    kz : numpy.ndarray
        Normal wavevector components with shape ``(n_wavelengths, n_layers)``.
    incident_angle_rad : float
        Incident angle in radians.

    Returns
    -------
    numpy.ndarray
        Cosine of propagation angles with shape ``(n_wavelengths, n_layers)``.
    """

    cos_theta = np.zeros_like(refractive_index, dtype=np.complex128)
    cos_theta[:, 0] = np.cos(incident_angle_rad)
    cos_theta[:, 1:] = kz[:, 1:] / (refractive_index[:, 1:] * k0[:, np.newaxis])
    return cos_theta


def compute_tm_scalar(
    refractive_index: ArrayC,
    k0: float,
    kz: ArrayC,
    thickness_m: ArrayF,
    incident_angle_rad: float,
    polarization: str,
) -> Tuple[ArrayC, ArrayC]:
    """Compute transfer matrix for one wavelength.

    Parameters
    ----------
    refractive_index : numpy.ndarray
        Refractive index values for each layer with shape ``(n_layers,)``.
    k0 : float
        Free-space wavevector magnitude for this wavelength.
    kz : numpy.ndarray
        Normal wavevector values for each layer with shape ``(n_layers,)``.
    thickness_m : numpy.ndarray
        Layer thicknesses in meters with shape ``(n_layers,)``.
    incident_angle_rad : float
        Incident angle in radians.
    polarization : str
        Polarization label (``"s"`` or ``"p"``).

    Returns
    -------
    tuple of numpy.ndarray
        Pair ``(tm, cos_theta)`` where ``tm`` has shape ``(2, 2)`` and
        ``cos_theta`` has shape ``(n_layers,)``.
    """

    n_layers = refractive_index.shape[0]
    cos_theta = np.zeros(n_layers, dtype=np.complex128)
    cos_theta[0] = np.cos(incident_angle_rad)
    cos_theta[1:] = kz[1:] / (refractive_index[1:] * k0)
    phi = kz * thickness_m

    _, tm = compute_dm_scalar(refractive_index[0], cos_theta[0], polarization)
    for i in range(1, n_layers - 1):
        dm, dim = compute_dm_scalar(refractive_index[i], cos_theta[i], polarization)
        pm = compute_pm_scalar(phi[i])
        tm = np.matmul(tm, dm)
        tm = np.matmul(tm, pm)
        tm = np.matmul(tm, dim)

    dm, _ = compute_dm_scalar(refractive_index[-1], cos_theta[-1], polarization)
    tm = np.matmul(tm, dm)
    return tm, cos_theta


def compute_tm_batch(
    refractive_index: ArrayC,
    k0: ArrayF,
    kz: ArrayC,
    thickness_m: ArrayF,
    incident_angle_rad: float,
    polarization: str,
) -> Tuple[ArrayC, ArrayC]:
    """Compute transfer matrices for all wavelengths in batch form.

    Parameters
    ----------
    refractive_index : numpy.ndarray
        Complex refractive index array with shape
        ``(number_of_wavelengths, number_of_layers)``.
    k0 : numpy.ndarray
        Free-space wavevector magnitudes with shape ``(number_of_wavelengths,)``.
    kz : numpy.ndarray
        Complex normal wavevector array with shape
        ``(number_of_wavelengths, number_of_layers)``.
    thickness_m : numpy.ndarray
        Layer thicknesses in meters with shape ``(number_of_layers,)``.
    incident_angle_rad : float
        Incident angle in radians.
    polarization : str
        Polarization label (``"s"`` or ``"p"``).

    Returns
    -------
    tuple of numpy.ndarray
        Pair ``(tm, cos_theta)`` where ``tm`` has shape
        ``(number_of_wavelengths, 2, 2)`` and ``cos_theta`` has shape
        ``(number_of_wavelengths, number_of_layers)``.
    """

    n_wavelengths, n_layers = refractive_index.shape
    phi = kz * thickness_m[np.newaxis, :]
    cos_theta = _compute_cos_theta(refractive_index, k0, kz, incident_angle_rad)

    tm = np.zeros((n_wavelengths, 2, 2), dtype=np.complex128)
    if polarization == "s":
        tm[:, 0, 0] = 1.0
        tm[:, 0, 1] = 1.0
        tm[:, 1, 0] = refractive_index[:, 0] * cos_theta[:, 0]
        tm[:, 1, 1] = -refractive_index[:, 0] * cos_theta[:, 0]
    else:
        tm[:, 0, 0] = cos_theta[:, 0]
        tm[:, 0, 1] = cos_theta[:, 0]
        tm[:, 1, 0] = refractive_index[:, 0]
        tm[:, 1, 1] = -refractive_index[:, 0]

    for i in range(1, n_layers - 1):
        dm = np.zeros((n_wavelengths, 2, 2), dtype=np.complex128)
        dim = np.zeros((n_wavelengths, 2, 2), dtype=np.complex128)

        if polarization == "s":
            dm[:, 0, 0] = 1.0
            dm[:, 0, 1] = 1.0
            dm[:, 1, 0] = refractive_index[:, i] * cos_theta[:, i]
            dm[:, 1, 1] = -refractive_index[:, i] * cos_theta[:, i]
        else:
            dm[:, 0, 0] = cos_theta[:, i]
            dm[:, 0, 1] = cos_theta[:, i]
            dm[:, 1, 0] = refractive_index[:, i]
            dm[:, 1, 1] = -refractive_index[:, i]

        det = 1.0 / (dm[:, 0, 0] * dm[:, 1, 1] - dm[:, 0, 1] * dm[:, 1, 0])
        dim[:, 0, 0] = det * dm[:, 1, 1]
        dim[:, 0, 1] = -det * dm[:, 0, 1]
        dim[:, 1, 0] = -det * dm[:, 1, 0]
        dim[:, 1, 1] = det * dm[:, 0, 0]

        pm = compute_pm_batch(phi[:, i])
        tm = np.matmul(tm, dm)
        tm = np.matmul(tm, pm)
        tm = np.matmul(tm, dim)

    dm = np.zeros((n_wavelengths, 2, 2), dtype=np.complex128)
    if polarization == "s":
        dm[:, 0, 0] = 1.0
        dm[:, 0, 1] = 1.0
        dm[:, 1, 0] = refractive_index[:, -1] * cos_theta[:, -1]
        dm[:, 1, 1] = -refractive_index[:, -1] * cos_theta[:, -1]
    else:
        dm[:, 0, 0] = cos_theta[:, -1]
        dm[:, 0, 1] = cos_theta[:, -1]
        dm[:, 1, 0] = refractive_index[:, -1]
        dm[:, 1, 1] = -refractive_index[:, -1]

    tm = np.matmul(tm, dm)
    return tm, cos_theta


def compute_transfer_matrix(
    refractive_index: ArrayC,
    k0: ArrayF,
    kz: ArrayC,
    thickness_m: ArrayF,
    incident_angle_rad: float,
    polarization: str,
    backend: str = "auto",
) -> Tuple[ArrayC, ArrayC]:
    """Dispatch transfer-matrix evaluation to scalar or vectorized backend.

    Parameters
    ----------
    refractive_index : numpy.ndarray
        Complex refractive-index array with shape ``(n_wavelengths, n_layers)``.
    k0 : numpy.ndarray
        Free-space wavevector magnitudes with shape ``(n_wavelengths,)``.
    kz : numpy.ndarray
        Complex normal wavevector values with shape ``(n_wavelengths, n_layers)``.
    thickness_m : numpy.ndarray
        Layer thickness array.
    incident_angle_rad : float
        Incident angle in radians.
    polarization : str
        Polarization label.
    backend : str, optional
        ``"vectorized"`` forces batched execution, ``"scalar"`` forces looping,
        and ``"auto"`` currently maps to vectorized execution.

    Returns
    -------
    tuple of numpy.ndarray
        Transfer matrix and cosine-angle arrays.
    """

    if backend not in {"auto", "scalar", "vectorized"}:
        raise ValueError("backend must be one of {'auto', 'scalar', 'vectorized'}")

    if backend in {"auto", "vectorized"}:
        return compute_tm_batch(
            refractive_index, k0, kz, thickness_m, incident_angle_rad, polarization
        )

    n_wavelengths = refractive_index.shape[0]
    tm = np.zeros((n_wavelengths, 2, 2), dtype=np.complex128)
    cos_theta = np.zeros_like(refractive_index, dtype=np.complex128)
    for i in range(n_wavelengths):
        tm[i], cos_theta[i] = compute_tm_scalar(
            refractive_index[i],
            k0[i],
            kz[i],
            thickness_m,
            incident_angle_rad,
            polarization,
        )
    return tm, cos_theta
