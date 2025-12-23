"""
This file contains Radial Symmetry to find the super-localization of a microbubble.
"""

from typing import Tuple

import numpy as np
from scipy.ndimage import uniform_filter


def radial_symmetry_center_3d(I: np.ndarray) -> Tuple[float, float, float, float]:
    """Radial symmetry to find super localization of microbubbles.

    Args:
        I (np.ndarray): The intensity matrix of the IQ.

    Returns:
        Tuple[float, float, float, float]: The coordinates in super resolution in pixels.
    """

    size_0, size_1, size_2 = I.shape

    # Create a meshgrid
    v0 = np.arange(-(size_0 - 1) / 2.0 + 0.5, (size_0 - 1) / 2.0 + 0.5)
    v1 = np.arange(-(size_1 - 1) / 2.0 + 0.5, (size_1 - 1) / 2.0 + 0.5)
    v2 = np.arange(-(size_2 - 1) / 2.0 + 0.5, (size_2 - 1) / 2.0 + 0.5)

    # Calculate derivatives along 45-degree shifted coordinates (u, v and w)
    dIdu = I[1:, 1:, 1:] - I[:-1, :-1, :-1]
    dIdv = I[1:, :-1, :-1] - I[:-1, 1:, 1:]
    dIdw = I[1:, 1:, :-1] - I[:-1, :-1, 1:]

    # Smooth the derivative with an averaging filter
    fdu = uniform_filter(dIdu, size=3, mode="constant")
    fdv = uniform_filter(dIdv, size=3, mode="constant")
    fdw = uniform_filter(dIdw, size=3, mode="constant")

    # Calculate derivative along x, y and z
    dId_1 = 0.5 * (fdv - fdu) * ((fdv - fdu) != 0) + 0.5 * (dIdv - dIdu) * (
        (dIdv - dIdu) != 0
    )
    dId_2 = 0.5 * (fdw - fdv) * ((fdw - fdv) != 0) + 0.5 * (dIdw - dIdv) * (
        (dIdw - dIdv) != 0
    )
    dId_0 = 0.5 * (fdu - fdw) * ((fdu - fdw) != 0) + 0.5 * (dIdu - dIdw) * (
        (dIdu - dIdw) != 0
    )

    # Compute the magnitude of the gradient
    mag_dI = fdu**2 + fdv**2 + fdw**2

    # Compute the sum of every matrix elements
    sdI3 = mag_dI.sum()

    # Compute a guess of bubble center position
    guess_0 = np.einsum("ijk,i->", mag_dI, v0) / sdI3
    guess_1 = np.einsum("ijk,j->", mag_dI, v1) / sdI3
    guess_2 = np.einsum("ijk,k->", mag_dI, v2) / sdI3

    # Compute the weights which are ||u_d||_(u, v, w) / ||pixPos-guessPos|| (v_dim1, v_dim2, v_dim3)
    W = (mag_dI / (dId_0**2 + dId_1**2 + dId_2**2)) / np.sqrt(
        (v0[:, np.newaxis, np.newaxis] - guess_0) ** 2
        + (v1[np.newaxis, :, np.newaxis] - guess_1) ** 2
        + (v2[np.newaxis, np.newaxis, :] - guess_2) ** 2
    )

    # Compute Omega matrix elements for each voxel
    # (See p.260 of Baptiste Heiles. 3D Ultrasound Localization Microscopy. Human health and pathology. Université Paris sciences et lettres, 2019. English.)
    Omega_11 = -2 * (dId_1**2 + dId_2**2)
    Omega_22 = -2 * (dId_2**2 + dId_0**2)
    Omega_33 = -2 * (dId_1**2 + dId_0**2)
    Omega_12 = 2 * dId_1 * dId_0
    Omega_13 = 2 * dId_2 * dId_0
    Omega_23 = 2 * dId_2 * dId_1

    # Compute the elements of matrix M which is also symmetric and has the shape 3x3
    M_11 = np.einsum("ijk,ijk->", W, Omega_11)
    M_12 = np.einsum("ijk,ijk->", W, Omega_12)
    M_13 = np.einsum("ijk,ijk->", W, Omega_13)
    M_22 = np.einsum("ijk,ijk->", W, Omega_22)
    M_23 = np.einsum("ijk,ijk->", W, Omega_23)
    M_33 = np.einsum("ijk,ijk->", W, Omega_33)

    # Compute the second member
    B = np.zeros((3, 1))
    B[0] = np.einsum(
        "ijk,ijk->",
        W,
        np.einsum("ijk,i->ijk", Omega_11, v0)
        + np.einsum("ijk,j->ijk", Omega_12, v1)
        + np.einsum("ijk,k->ijk", Omega_13, v2),
    )
    B[1] = np.einsum(
        "ijk,ijk->",
        W,
        np.einsum("ijk,i->ijk", Omega_12, v0)
        + np.einsum("ijk,j->ijk", Omega_22, v1)
        + np.einsum("ijk,k->ijk", Omega_23, v2),
    )
    B[2] = np.einsum(
        "ijk,ijk->",
        W,
        np.einsum("ijk,i->ijk", Omega_13, v0)
        + np.einsum("ijk,j->ijk", Omega_23, v1)
        + np.einsum("ijk,k->ijk", Omega_33, v2),
    )

    # Compute the inverse of matrix M
    alpha = M_22 * M_33 - M_23**2
    delta = M_11 * M_33 - M_13**2
    phi = M_11 * M_22 - M_12**2
    beta = M_12 * M_33 - M_13 * M_23
    gamma = M_12 * M_23 - M_13 * M_22
    epsilon = M_11 * M_23 - M_13 * M_12
    Minv = np.array(
        [[alpha, -beta, gamma], [-beta, delta, -epsilon], [gamma, -epsilon, phi]]
    )
    Minv /= M_11 * alpha - M_12 * beta + M_13 * gamma

    # Compute the optimized bubble center position
    superloc_0, superloc_1, superloc_2 = Minv @ B

    # If radial symmetry failed because of NaN values, return weighted average sub-localization.
    if (
        np.isnan(superloc_0.item())
        or np.isnan(superloc_1.item())
        or np.isnan(superloc_2.item())
    ):
        return guess_0, guess_1, guess_2
    return superloc_0.item(), superloc_1.item(), superloc_2.item()
