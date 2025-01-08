import gc
from datetime import datetime

import matplotlib.colors as mcolors
import matplotlib.pylab as plt
import numpy as np
from loguru import logger

from tools.settings import Settings
from tools.technology import create_directory, images_to_video, clear_folder

colors = [
    "#f4f0e7",
    "#e74c3c",  # Vibrant red
    "#f39c12",  # Vibrant orange
    "#27ae60",  # Vibrant green
    "#2980b9",  # Vibrant blue
    "#8e44ad",  # Vibrant purple
    '#001426',
]  # Red, Green, Blue, Yellow, Cyan

cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=1000)


def generate():
    settings = Settings()
    filename = settings.filename
    create_directory(f"outputs/{filename}")
    clear_folder(f"outputs/{filename}")
    logger.info(f"Starting calculation")

    n = 120
    x = np.random.uniform(-1, 1, (n, n)) + np.random.uniform(-1, 1, (n, n)) * 1j
    v1 = np.random.uniform(-1, 1, n) + np.random.uniform(-1, 1, n) * 1j
    v2_ = np.random.uniform(-1, 1, n) + np.random.uniform(-1, 1, n) * 1j
    v2 = make_orthogonal(v1, v2_)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    for theta in np.linspace(0, 2 * np.pi, 300):
        fig, ax = plt.subplots(figsize=(12, 12), dpi=200)
        ax = fig.add_axes((0, 0, 1, 1), facecolor="#000D1E")
        ax.set_xticks([])
        ax.set_yticks([])

        U = np.eye(n, dtype=complex)

        U[:2, :2] = np.array(
            [
                [np.cos(theta), -1j * np.sin(theta)],
                [np.cos(-theta), -1j * np.sin(-theta)],
            ]
        )

        # Rotate the matrix x using the unitary matrix U
        x_rotated = U @ x @ U.conj().T

        eigenvalues, eigenvectors = np.linalg.eig(x_rotated)
        norms = np.abs(eigenvalues)
        ax.scatter(
            eigenvalues.real,
            eigenvalues.imag,
            color=[cmap(z / max(norms)) for z in norms],
            s=420,
            lw=0,
            alpha=0.85,
        )
        ax.set_xlim(-7, 7)
        ax.set_ylim(-7, 7)
        time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        logger.info(f"{time_string}.png Saved")
        fig.savefig(f'outputs/{filename}/{time_string}.png', facecolor="#f4f0e7")
        plt.close()
        gc.collect()
    images_to_video(f'outputs/{filename}', f'{filename}.mp4', 60)
    logger.info(f"Finished")


def rotation_matrix(u, v, theta):
    """
    Generate a rotation matrix that rotates the span of two orthonormal vectors u and v by an angle theta.

    Parameters:
        u (numpy.ndarray): Orthonormal vector (n-dimensional).
        v (numpy.ndarray): Orthonormal vector (n-dimensional).
        theta (float): Angle of rotation in radians.

    Returns:
        numpy.ndarray: The rotation matrix.
    """
    u = np.asarray(u).reshape(-1, 1)  # Ensure column vector
    v = np.asarray(v).reshape(-1, 1)  # Ensure column vector

    # Outer products
    uuT = u @ u.T
    vvT = v @ v.T
    uvT = u @ v.T
    vuT = v @ u.T

    # Identity matrix
    n = len(u)
    I = np.eye(n)

    # Rotation matrix
    A = I + np.sin(theta) * (vuT - uvT) + (np.cos(theta) - 1) * (uuT + vvT)

    return A


def make_orthogonal(v1, v2):
    """
    Adjust v2 to make it orthogonal to v1 using the Gram-Schmidt process.

    Parameters:
        v1 (numpy.ndarray): The first vector.
        v2 (numpy.ndarray): The second vector to be adjusted.

    Returns:
        numpy.ndarray: The adjusted v2 that is orthogonal to v1.
    """
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)

    # Normalize v1 to ensure a proper orthogonal basis
    v1 = v1 / np.linalg.norm(v1)

    # Project v2 onto v1
    projection = np.dot(v2, v1) * v1

    # Subtract the projection from v2 to make it orthogonal to v1
    v2_orthogonal = v2 - projection

    return v2_orthogonal
