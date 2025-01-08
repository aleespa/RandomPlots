import gc
import os
import sys
from datetime import datetime

import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

matplotlib.use('Agg')


def vectorized_sample_complex_pairs(sample_size: int):
    # Sample 2n random angles from 0 to 2*pi (2 for each pair)
    np.random.seed(1)
    thetas = np.random.uniform(0, 2 * np.pi, size=2 * sample_size)

    # Compute complex numbers
    zs = np.exp(1j * thetas)

    # Reshape to get n pairs
    pairs = zs.reshape(sample_size, 2)

    return pairs


def calculate_matrix_v1(t):
    return np.array([[t[0], 1j], [0.5, t[1]]])


def calculate_matrix_v2(t):
    return np.array([[t[0], 1j], [-0.5, t[1]]])


def calculate_eigenvalues(x: np.array):
    return np.linalg.eigvals(x)


def generate_plot(x, y, directory, color):
    current_time = datetime.now()
    time_string = current_time.strftime('%Y-%m-%d_%H-%M-%S-%f')
    fig, _ = plt.subplots(figsize=(9, 16), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1], facecolor='#f4f0e7')
    ax.scatter(x[0], y[1], s=5, lw=0, color=color[0])
    ax.scatter(x[1], y[0], s=5, lw=0, color=color[1])
    y1, y2 = -2, 2
    x1, x2 = -2, 2
    w = x2 - x1
    h = y2 - y1
    z = (16 / 18) * w - (1 / 2) * h
    ax.set_xlim(x1, x2)
    ax.set_ylim(y1 - z, y2 + z)
    if not os.path.exists(f"outputs/{directory}"):
        os.makedirs(f"outputs/{directory}")
    fig.savefig(f'outputs/{directory}/{time_string}.png', facecolor='k')
    plt.close()
    del x, y


def images_to_video(image_folder, video_name, fps):
    """
    Convert a folder of images into an MP4 video.

    Parameters:
    - image_folder: Folder containing image frames.
    - video_name: Name of the output video file (including .mp4 extension).
    - fps: Frames per second for the output video.
    """
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()  # Make sure the images are in the correct order

    # Get the size of the images
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    h, w, layers = frame.shape
    size = (w, h)

    # Create a video writer object
    out = cv2.VideoWriter(
        "outputs/" + video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, size
    )

    for image in images:
        img_path = os.path.join(image_folder, image)
        img = cv2.imread(img_path)
        out.write(img)

    out.release()


def generate():
    filename = sys.argv[1]
    sample_size = 5000
    sample = vectorized_sample_complex_pairs(sample_size)
    for i, r in enumerate(np.linspace(0, 4.8, 360)):
        Z1 = np.array(
            [calculate_eigenvalues(calculate_matrix_v1(t)) for t in sample * r]
        ).ravel()
        Z2 = np.array(
            [calculate_eigenvalues(calculate_matrix_v2(t)) for t in sample * r]
        ).ravel()
        generate_plot(
            [Z1.real, Z2.real], [Z1.imag, Z2.imag], filename, ['#cc0000', '#000033']
        )
        gc.collect()
    images_to_video(f"outputs/{filename}", f'{filename}.mp4', 30)
