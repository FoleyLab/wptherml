"""Shared transfer-matrix kernels for stratified media."""

import numpy as np


def batch_matmul(a, b):
    """Multiply stacks of 2x2 matrices with NumPy broadcasting."""
    return np.einsum("...ij,...jk->...ik", a, b)


def compute_dm(refractive_index, cosine_theta, polarization):
    """Compute dynamic matrices and their inverses for one or more layers."""
    refractive_index = np.asarray(refractive_index, dtype=np.complex128)
    cosine_theta = np.asarray(cosine_theta, dtype=np.complex128)

    shape = np.broadcast_shapes(refractive_index.shape, cosine_theta.shape) + (2, 2)
    refractive_index = np.broadcast_to(refractive_index, shape[:-2])
    cosine_theta = np.broadcast_to(cosine_theta, shape[:-2])

    dm = np.zeros(shape, dtype=np.complex128)
    dim = np.zeros_like(dm)

    if polarization == "s":
        dm[..., 0, 0] = 1
        dm[..., 0, 1] = 1
        dm[..., 1, 0] = refractive_index * cosine_theta
        dm[..., 1, 1] = -refractive_index * cosine_theta
    elif polarization == "p":
        dm[..., 0, 0] = cosine_theta
        dm[..., 0, 1] = cosine_theta
        dm[..., 1, 0] = refractive_index
        dm[..., 1, 1] = -refractive_index
    else:
        raise ValueError(f"polarization must be 's' or 'p', got {polarization!r}")

    det_inv = 1 / (dm[..., 0, 0] * dm[..., 1, 1] - dm[..., 0, 1] * dm[..., 1, 0])
    dim[..., 0, 0] = det_inv * dm[..., 1, 1]
    dim[..., 0, 1] = -det_inv * dm[..., 0, 1]
    dim[..., 1, 0] = -det_inv * dm[..., 1, 0]
    dim[..., 1, 1] = det_inv * dm[..., 0, 0]

    return dm, dim


def compute_pm(phi):
    """Compute propagation matrices for one or more phase thicknesses."""
    phi = np.asarray(phi, dtype=np.complex128)
    pm = np.zeros(phi.shape + (2, 2), dtype=np.complex128)
    pm[..., 0, 0] = np.exp(-1j * phi)
    pm[..., 1, 1] = np.exp(1j * phi)
    return pm


def compute_pm_gradient(kz, phi):
    """Compute dP/dd for one or more layer phase thicknesses."""
    kz = np.asarray(kz, dtype=np.complex128)
    phi = np.asarray(phi, dtype=np.complex128)
    shape = np.broadcast_shapes(kz.shape, phi.shape)
    kz = np.broadcast_to(kz, shape)
    phi = np.broadcast_to(phi, shape)

    pm_gradient = np.zeros(shape + (2, 2), dtype=np.complex128)
    pm_gradient[..., 0, 0] = -1j * kz * np.exp(-1j * phi)
    pm_gradient[..., 1, 1] = 1j * kz * np.exp(1j * phi)
    return pm_gradient


def compute_angles(refractive_index, k0, kz, incident_angle):
    """Compute layer angles and cosines from refractive indices and kz."""
    refractive_index = np.asarray(refractive_index, dtype=np.complex128)
    kz = np.asarray(kz, dtype=np.complex128)
    k0 = np.asarray(k0)
    incident_angle = np.asarray(incident_angle)

    cos_theta = np.zeros_like(kz, dtype=np.complex128)
    cos_theta[..., 0] = np.cos(incident_angle)
    cos_theta[..., 1:] = kz[..., 1:] / (refractive_index[..., 1:] * k0[..., np.newaxis])

    theta = np.zeros_like(kz, dtype=np.complex128)
    theta[..., 0] = incident_angle
    theta[..., 1:] = np.arccos(cos_theta[..., 1:])
    return theta, cos_theta


def _validate_gradient_layers(gradient_layers, number_of_layers):
    gradient_layers = np.asarray(gradient_layers, dtype=int)
    if gradient_layers.ndim == 0:
        gradient_layers = gradient_layers[np.newaxis]
    if np.any((gradient_layers < 1) | (gradient_layers > number_of_layers - 2)):
        raise ValueError(
            "gradient layers must refer to finite-thickness internal layers "
            f"1 through {number_of_layers - 2}"
        )
    return gradient_layers


def compute_transfer_matrix(
    refractive_index, k0, kz, thickness, incident_angle, polarization
):
    """Compute transfer matrices over any leading wavelength/angle dimensions."""
    refractive_index = np.asarray(refractive_index, dtype=np.complex128)
    kz = np.asarray(kz, dtype=np.complex128)
    thickness = np.asarray(thickness)
    number_of_layers = refractive_index.shape[-1]

    theta, cos_theta = compute_angles(refractive_index, k0, kz, incident_angle)
    phi = kz * thickness
    pm = compute_pm(phi)
    dm, dim = compute_dm(refractive_index, cos_theta, polarization)

    matrix_shape = refractive_index.shape[:-1] + (2, 2)
    transfer_matrix = np.broadcast_to(np.eye(2, dtype=np.complex128), matrix_shape).copy()
    transfer_matrix = batch_matmul(transfer_matrix, dim[..., 0, :, :])

    for layer in range(1, number_of_layers - 1):
        transfer_matrix = batch_matmul(transfer_matrix, dm[..., layer, :, :])
        transfer_matrix = batch_matmul(transfer_matrix, pm[..., layer, :, :])
        transfer_matrix = batch_matmul(transfer_matrix, dim[..., layer, :, :])

    transfer_matrix = batch_matmul(transfer_matrix, dm[..., -1, :, :])
    return transfer_matrix, theta, cos_theta


def compute_transfer_matrices(
    refractive_index, k0, kz, thickness, incident_angle, polarizations
):
    """Compute transfer matrices for a list of polarizations."""
    matrices = []
    theta = None
    cos_theta = None
    for polarization in polarizations:
        matrix, theta, cos_theta = compute_transfer_matrix(
            refractive_index, k0, kz, thickness, incident_angle, polarization
        )
        matrices.append(matrix)
    return np.stack(matrices, axis=-3), theta, cos_theta


def compute_transfer_matrix_gradients(
    refractive_index,
    k0,
    kz,
    thickness,
    incident_angle,
    polarization,
    gradient_layers,
):
    """Compute transfer matrices and dM/dd for selected layer thicknesses."""
    refractive_index = np.asarray(refractive_index, dtype=np.complex128)
    kz = np.asarray(kz, dtype=np.complex128)
    thickness = np.asarray(thickness)
    number_of_layers = refractive_index.shape[-1]
    gradient_layers = _validate_gradient_layers(gradient_layers, number_of_layers)
    number_of_gradients = len(gradient_layers)

    theta, cos_theta = compute_angles(refractive_index, k0, kz, incident_angle)
    phi = kz * thickness
    pm = compute_pm(phi)
    dm, dim = compute_dm(refractive_index, cos_theta, polarization)

    prefix_shape = refractive_index.shape[:-1]
    matrix_shape = prefix_shape + (2, 2)
    gradient_shape = prefix_shape + (number_of_gradients, 2, 2)
    transfer_matrix = np.broadcast_to(np.eye(2, dtype=np.complex128), matrix_shape).copy()
    transfer_matrix = batch_matmul(transfer_matrix, dim[..., 0, :, :])
    transfer_matrix_gradient = np.broadcast_to(
        dim[..., 0, :, :][..., np.newaxis, :, :], gradient_shape
    ).copy()

    gradient_axis_shape = (1,) * len(prefix_shape) + (number_of_gradients, 1, 1)
    for layer in range(1, number_of_layers - 1):
        dm_layer = dm[..., layer, :, :]
        dim_layer = dim[..., layer, :, :]
        pm_layer = pm[..., layer, :, :]
        pm_gradient_layer = compute_pm_gradient(kz[..., layer], phi[..., layer])

        transfer_matrix = batch_matmul(transfer_matrix, dm_layer)
        transfer_matrix = batch_matmul(transfer_matrix, pm_layer)
        transfer_matrix = batch_matmul(transfer_matrix, dim_layer)

        transfer_matrix_gradient = batch_matmul(
            transfer_matrix_gradient, dm_layer[..., np.newaxis, :, :]
        )
        layer_matches = (gradient_layers == layer).reshape(gradient_axis_shape)
        pm_for_gradient = np.where(
            layer_matches,
            pm_gradient_layer[..., np.newaxis, :, :],
            pm_layer[..., np.newaxis, :, :],
        )
        transfer_matrix_gradient = batch_matmul(transfer_matrix_gradient, pm_for_gradient)
        transfer_matrix_gradient = batch_matmul(
            transfer_matrix_gradient, dim_layer[..., np.newaxis, :, :]
        )

    transfer_matrix = batch_matmul(transfer_matrix, dm[..., -1, :, :])
    transfer_matrix_gradient = batch_matmul(
        transfer_matrix_gradient, dm[..., -1, :, :][..., np.newaxis, :, :]
    )

    return transfer_matrix, transfer_matrix_gradient, theta, cos_theta


def compute_spectrum_from_transfer_matrix(transfer_matrix, refractive_index, cos_theta):
    """Compute R, T, and emissivity from transfer matrices."""
    r = transfer_matrix[..., 1, 0] / transfer_matrix[..., 0, 0]
    t = 1 / transfer_matrix[..., 0, 0]
    factor = (
        refractive_index[..., -1]
        * cos_theta[..., -1]
        / (refractive_index[..., 0] * cos_theta[..., 0])
    )
    while factor.ndim < r.ndim:
        factor = factor[..., np.newaxis]
    reflectivity = np.real(r * np.conj(r))
    transmissivity = np.real(t * np.conj(t) * factor)
    emissivity = 1 - reflectivity - transmissivity
    return reflectivity, transmissivity, emissivity


def compute_spectrum_gradients_from_transfer_matrix(
    transfer_matrix, transfer_matrix_gradient, refractive_index, cos_theta
):
    """Compute dR/dd, dT/dd, and dE/dd from dM/dd."""
    r = transfer_matrix[..., 1, 0] / transfer_matrix[..., 0, 0]
    t = 1 / transfer_matrix[..., 0, 0]
    factor = (
        refractive_index[..., -1]
        * cos_theta[..., -1]
        / (refractive_index[..., 0] * cos_theta[..., 0])
    )

    m00 = transfer_matrix[..., 0, 0][..., np.newaxis]
    m10 = transfer_matrix[..., 1, 0][..., np.newaxis]
    dm00 = transfer_matrix_gradient[..., 0, 0]
    dm10 = transfer_matrix_gradient[..., 1, 0]

    r_prime = (m00 * dm10 - m10 * dm00) / (m00**2)
    t_prime = -dm00 / (m00**2)
    r = r[..., np.newaxis]
    t = t[..., np.newaxis]
    factor = factor[..., np.newaxis]

    reflectivity_gradient = np.real(r_prime * np.conj(r) + r * np.conj(r_prime))
    transmissivity_gradient = np.real(
        (t_prime * np.conj(t) + t * np.conj(t_prime)) * factor
    )
    emissivity_gradient = -transmissivity_gradient - reflectivity_gradient
    return reflectivity_gradient, transmissivity_gradient, emissivity_gradient
