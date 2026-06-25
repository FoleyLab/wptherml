"""High-performance vectorized transfer-matrix kernels.

This module is the optimized evaluation path for the refactored architecture.
It is mathematically identical to the ``tmm_core`` dynamical-matrix formulation
but is restructured for speed along the dimensions that dominate cost:

* **Wavelengths** (``N_wl``) are almost always the largest dimension and are
  fully vectorized as a batch axis.
* **Layers** (``N_layers``) set the length of the sequential matrix product and
  are the one axis that cannot be parallelized -- so the per-layer work is made
  as cheap as possible.
* **Angles** (``N_angle``) and **polarizations** (``N_pol`` <= 2) are small and
  are carried as leading batch axes so there is no Python-level loop over them.
* **Replicas** (``N_replica``) -- independent structures that differ only in
  their layer thicknesses -- are carried as one more leading batch axis, so an
  entire ensemble (e.g. a Monte-Carlo swarm over thickness uncertainty) is
  evaluated in a single vectorized pass with no Python loop over replicas.

The internal working layout for every 2x2 matrix is four component arrays of
shape ``(N_pol, N_replica, N_wl, N_angle)``. Interface matrices do not depend on
thickness and so broadcast across the replica axis; only the propagation factors
carry per-replica thickness.

Three ideas drive the speedup relative to a generic ``einsum`` of ``(...,2,2)``
matrices:

1. **Interface (Fresnel) matrices.** The dynamical-matrix product
   ``D0^-1 (D1 P1 D1^-1) ... D_{N-1}`` regroups exactly into
   ``I0 P1 I1 P2 ... I_{N-2}`` where ``I_l = D_l^-1 D_{l+1}`` is the interface
   matrix between adjacent layers. This removes every explicit matrix inverse
   and roughly halves the number of full 2x2 multiplies in the chain. Because
   ``I_l`` is computed as exactly ``D_l^-1 D_{l+1}`` the result is bitwise
   equivalent to the dynamical-matrix kernels (no Fresnel sign-convention drift).

2. **Component arrays instead of (...,2,2) tensors.** The four fused
   multiply-adds of a 2x2 product are written out by hand, avoiding
   ``einsum``/``matmul`` dispatch overhead and the strided access of a trailing
   ``(2, 2)`` block.

3. **Diagonal propagation as row/column scaling.** ``P_m`` is diagonal, so a
   product with ``P_m`` is just a scaling of two rows or columns -- two
   multiplies per element rather than a full matrix product.

Gradients with respect to layer thicknesses use the adjoint (prefix/suffix
product) method, computing every requested ``dM/dd_l`` in ``O(N_layers)`` total
work rather than recomputing the whole chain once per gradient layer.
"""

from __future__ import annotations

import numpy as np


def _normalize_polarizations(polarizations):
    if polarizations is None:
        return ["p"]
    if isinstance(polarizations, str):
        polarizations = [polarizations]
    polarization_list = [p.lower() for p in polarizations]
    invalid = [p for p in polarization_list if p not in {"s", "p"}]
    if invalid:
        raise ValueError(f"polarizations must be 's' or 'p', got {invalid}")
    return polarization_list


def _layer_geometry(refractive_indices, wavelengths, angles):
    """Return (n, kz, cos_theta), each shape (N_layers, N_wl, N_angle).

    ``cos_theta`` for the incident layer is set to ``cos(angle)`` exactly, to
    match the reference dynamical-matrix kernels to floating-point precision.
    """
    refractive_indices = np.asarray(refractive_indices, dtype=np.complex128)
    wavelengths = np.asarray(wavelengths, dtype=float)
    angles = np.atleast_1d(np.asarray(angles, dtype=float))

    n = np.transpose(refractive_indices)[:, :, np.newaxis]  # (N_layers, N_wl, 1)
    k0 = (2.0 * np.pi / wavelengths)[np.newaxis, :, np.newaxis]  # (1, N_wl, 1)
    sin_angle = np.sin(angles)[np.newaxis, np.newaxis, :]  # (1, 1, N_angle)

    n0 = n[0:1]  # incident-layer index
    kx = n0 * sin_angle * k0  # (1, N_wl, N_angle), conserved across layers
    kz = np.sqrt((n * k0) ** 2 - kx**2)  # (N_layers, N_wl, N_angle)

    cos_theta = kz / (n * k0)
    cos_theta[0] = np.cos(angles)[np.newaxis, :]  # exact incident-layer cosine
    return n, kz, cos_theta


def _interface_components(n, cos_theta, polarizations):
    """Stacked interface-matrix components, shape (N_pol, N_interfaces, N_wl, N_angle).

    Each interface matrix is ``[[i_sum, i_dif], [i_dif, i_sum]]`` with
    ``i_sum = p + q``, ``i_dif = p - q``, ``p = A_{l+1}/(2 A_l)``,
    ``q = B_{l+1}/(2 B_l)`` and the dynamical-matrix parameters
    ``D = [[A, A], [B, -B]]``:

        s-pol: A = 1,         B = n cos_theta
        p-pol: A = cos_theta, B = n
    """
    n_b = np.broadcast_to(n, cos_theta.shape)
    a_stack, b_stack = [], []
    for polarization in polarizations:
        if polarization == "s":
            a_stack.append(np.ones_like(cos_theta))
            b_stack.append(n_b * cos_theta)
        else:  # "p"
            a_stack.append(cos_theta)
            b_stack.append(n_b)
    A = np.stack(a_stack, axis=0)  # (N_pol, N_layers, N_wl, N_angle)
    B = np.stack(b_stack, axis=0)

    p = A[:, 1:] / (2.0 * A[:, :-1])
    q = B[:, 1:] / (2.0 * B[:, :-1])
    return p + q, p - q  # (N_pol, N_interfaces, N_wl, N_angle)


def _phases(kz, thicknesses, number_of_layers):
    """Propagation phases for internal layers.

    ``thicknesses`` may be 1-D ``(N_layers,)`` or 2-D ``(N_replica, N_layers)``.
    Returns ``phase`` with shape ``(N_internal, N_replica, N_wl, N_angle)`` and
    ``kz_internal`` with shape ``(N_internal, N_wl, N_angle)``.
    """
    thicknesses = np.asarray(thicknesses, dtype=float)
    if thicknesses.ndim == 1:
        thicknesses = thicknesses[np.newaxis, :]  # (1, N_layers)

    kz_internal = kz[1 : number_of_layers - 1]  # (N_internal, N_wl, N_angle)
    d_internal = thicknesses[:, 1 : number_of_layers - 1]  # (N_replica, N_internal)
    # -> (N_internal, N_replica, 1, 1) * (N_internal, 1, N_wl, N_angle)
    phase = (
        d_internal.T[:, :, np.newaxis, np.newaxis]
        * kz_internal[:, np.newaxis, :, :]
    )
    return phase, kz_internal


def _matmul_components(a, b):
    """Full 2x2 product of two component-tuple matrices (broadcasting)."""
    a00, a01, a10, a11 = a
    b00, b01, b10, b11 = b
    return (
        a00 * b00 + a01 * b10,
        a00 * b01 + a01 * b11,
        a10 * b00 + a11 * b10,
        a10 * b01 + a11 * b11,
    )


def _scale_columns(m, c0, c1):
    """Right-multiply a component-tuple matrix by diag(c0, c1) (scales columns)."""
    m00, m01, m10, m11 = m
    return (m00 * c0, m01 * c1, m10 * c0, m11 * c1)


def _scale_rows(m, r0, r1):
    """Left-multiply a component-tuple matrix by diag(r0, r1) (scales rows)."""
    m00, m01, m10, m11 = m
    return (m00 * r0, m01 * r0, m10 * r1, m11 * r1)


def _interface(i_sum, i_dif, index):
    """Interface matrix at ``index`` with a singleton replica axis inserted.

    Returns components of shape ``(N_pol, 1, N_wl, N_angle)`` so they broadcast
    against working matrices of shape ``(N_pol, N_replica, N_wl, N_angle)``.
    """
    s = i_sum[:, index][:, np.newaxis]
    d = i_dif[:, index][:, np.newaxis]
    return (s, d, d, s)


def _transmission_factor(n, cos_theta):
    """Real prefactor (n_last cos_last)/(n0 cos0), shape (1, 1, N_wl, N_angle)."""
    factor = (n[-1] * cos_theta[-1]) / (n[0] * cos_theta[0])  # (N_wl, N_angle)
    return factor[np.newaxis, np.newaxis]


def _spectra_from_matrix(matrix, factor):
    m00, _m01, m10, _m11 = matrix
    r = m10 / m00
    t = 1.0 / m00
    reflectivity = np.real(r * np.conj(r))
    transmissivity = np.real(t * np.conj(t) * factor)
    emissivity = 1.0 - reflectivity - transmissivity
    return r, t, reflectivity, transmissivity, emissivity


def _forward_chain(i_sum, i_dif, exp_minus, exp_plus, number_of_internal):
    """Evaluate M = I_0 P_1 I_1 ... I_{N-2} as component arrays."""
    matrix = _interface(i_sum, i_dif, 0)
    for j in range(number_of_internal):
        matrix = _scale_columns(matrix, exp_minus[j], exp_plus[j])
        matrix = _matmul_components(matrix, _interface(i_sum, i_dif, j + 1))
    return matrix


def _exp_factors(phase):
    """Propagation factors with a leading polarization axis for broadcasting.

    ``phase`` has shape ``(N_internal, N_replica, N_wl, N_angle)``; the returned
    arrays have shape ``(N_internal, 1, N_replica, N_wl, N_angle)`` so that
    ``exp[j]`` broadcasts against ``(N_pol, N_replica, N_wl, N_angle)``.
    """
    exp_minus = np.exp(-1j * phase)[:, np.newaxis]
    exp_plus = np.exp(1j * phase)[:, np.newaxis]
    return exp_minus, exp_plus


def _squeeze_replica(array, had_replica, axis):
    """Drop the replica axis when the caller passed a single (1-D) thickness."""
    if had_replica:
        return array
    return np.squeeze(array, axis=axis)


def solve_rt(
    refractive_indices, wavelengths, angles, thicknesses, polarizations=None
):
    """Compute reflectivity, transmissivity and emissivity.

    ``thicknesses`` may be 1-D ``(N_layers,)`` for a single structure or 2-D
    ``(N_replica, N_layers)`` for an ensemble.

    Returns three arrays of shape ``(N_wl, N_angle, N_pol)`` for a single
    structure, or ``(N_replica, N_wl, N_angle, N_pol)`` for an ensemble.
    """
    polarizations = _normalize_polarizations(polarizations)
    had_replica = np.asarray(thicknesses, dtype=float).ndim == 2

    n, kz, cos_theta = _layer_geometry(refractive_indices, wavelengths, angles)
    number_of_layers = n.shape[0]
    i_sum, i_dif = _interface_components(n, cos_theta, polarizations)
    phase, _ = _phases(kz, thicknesses, number_of_layers)
    exp_minus, exp_plus = _exp_factors(phase)
    number_of_internal = phase.shape[0]

    matrix = _forward_chain(i_sum, i_dif, exp_minus, exp_plus, number_of_internal)
    factor = _transmission_factor(n, cos_theta)
    _r, _t, reflectivity, transmissivity, emissivity = _spectra_from_matrix(
        matrix, factor
    )

    # (N_pol, N_replica, N_wl, N_angle) -> (N_replica, N_wl, N_angle, N_pol)
    out = []
    for spectrum in (reflectivity, transmissivity, emissivity):
        spectrum = np.moveaxis(spectrum, 0, -1)  # replica, wl, angle, pol
        spectrum = _squeeze_replica(spectrum, had_replica, axis=0)
        out.append(spectrum)
    return tuple(out)


def _validate_gradient_layers(gradient_layers, number_of_layers):
    gradient_layers = np.atleast_1d(np.asarray(gradient_layers, dtype=int))
    if np.any((gradient_layers < 1) | (gradient_layers > number_of_layers - 2)):
        raise ValueError(
            "gradient layers must refer to finite-thickness internal layers "
            f"1 through {number_of_layers - 2}"
        )
    return gradient_layers


def solve_rt_gradients(
    refractive_indices,
    wavelengths,
    angles,
    thicknesses,
    gradient_layers,
    polarizations=None,
):
    """Spectra and their layer-thickness gradients via the adjoint method.

    ``thicknesses`` may be 1-D ``(N_layers,)`` or 2-D ``(N_replica, N_layers)``.

    Returns
    -------
    spectra : tuple of ndarray
        ``(R, T, E)``; shape ``(N_wl, N_angle, N_pol)`` (single structure) or
        ``(N_replica, N_wl, N_angle, N_pol)`` (ensemble).
    gradients : tuple of ndarray
        ``(dR_dd, dT_dd, dE_dd)``; shape ``(N_wl, N_angle, N_pol, N_grad)`` or
        ``(N_replica, N_wl, N_angle, N_pol, N_grad)``.
    """
    polarizations = _normalize_polarizations(polarizations)
    had_replica = np.asarray(thicknesses, dtype=float).ndim == 2

    n, kz, cos_theta = _layer_geometry(refractive_indices, wavelengths, angles)
    number_of_layers = n.shape[0]
    gradient_layers = _validate_gradient_layers(gradient_layers, number_of_layers)

    i_sum, i_dif = _interface_components(n, cos_theta, polarizations)
    phase, kz_internal = _phases(kz, thicknesses, number_of_layers)
    exp_minus, exp_plus = _exp_factors(phase)
    number_of_internal = phase.shape[0]

    # Prefix products. prefix[j] (internal layer m = j+1) holds everything left
    # of P_m: I_0 P_1 I_1 ... P_{m-1} I_{m-1}.   prefix[0] = I_0.
    prefix = [None] * number_of_internal
    running = _interface(i_sum, i_dif, 0)
    prefix[0] = running
    for j in range(1, number_of_internal):
        running = _scale_columns(running, exp_minus[j - 1], exp_plus[j - 1])
        running = _matmul_components(running, _interface(i_sum, i_dif, j))
        prefix[j] = running

    # Suffix products. suffix[j] holds everything right of P_m:
    # I_m P_{m+1} I_{m+1} ... I_{N-2}.   suffix[last] = I_{N-2}.
    suffix = [None] * number_of_internal
    suffix[number_of_internal - 1] = _interface(i_sum, i_dif, number_of_internal)
    for j in range(number_of_internal - 2, -1, -1):
        propagated = _scale_rows(suffix[j + 1], exp_minus[j + 1], exp_plus[j + 1])
        suffix[j] = _matmul_components(_interface(i_sum, i_dif, j + 1), propagated)

    # Full matrix: M = prefix[0] P_1 suffix[0].
    m_full = _matmul_components(
        _scale_columns(prefix[0], exp_minus[0], exp_plus[0]), suffix[0]
    )
    factor = _transmission_factor(n, cos_theta)
    r, t, reflectivity, transmissivity, emissivity = _spectra_from_matrix(
        m_full, factor
    )
    m00 = m_full[0]
    m10 = m_full[2]
    m00_sq = m00 * m00

    dR_list, dT_list, dE_list = [], [], []
    for layer in gradient_layers:
        j = layer - 1  # internal index
        kz_j = kz_internal[j]  # (N_wl, N_angle)
        dP0 = -1j * kz_j * exp_minus[j]
        dP1 = 1j * kz_j * exp_plus[j]
        dM = _matmul_components(_scale_columns(prefix[j], dP0, dP1), suffix[j])
        dm00, dm10 = dM[0], dM[2]
        r_prime = (m00 * dm10 - m10 * dm00) / m00_sq
        t_prime = -dm00 / m00_sq
        dR = np.real(r_prime * np.conj(r) + r * np.conj(r_prime))
        dT = np.real((t_prime * np.conj(t) + t * np.conj(t_prime)) * factor)
        dR_list.append(dR)
        dT_list.append(dT)
        dE_list.append(-dR - dT)

    # Spectra: (N_pol, N_replica, N_wl, N_angle) -> (..., N_pol)
    spectra = []
    for spectrum in (reflectivity, transmissivity, emissivity):
        spectrum = np.moveaxis(spectrum, 0, -1)
        spectrum = _squeeze_replica(spectrum, had_replica, axis=0)
        spectra.append(spectrum)

    # Gradients: stack -> (N_pol, N_replica, N_wl, N_angle, N_grad) -> (..., N_pol, N_grad)
    gradients = []
    for grad_list in (dR_list, dT_list, dE_list):
        grad = np.stack(grad_list, axis=-1)  # (N_pol, N_replica, N_wl, N_angle, N_grad)
        grad = np.moveaxis(grad, 0, -2)  # (N_replica, N_wl, N_angle, N_pol, N_grad)
        grad = _squeeze_replica(grad, had_replica, axis=0)
        gradients.append(grad)

    return tuple(spectra), tuple(gradients)
