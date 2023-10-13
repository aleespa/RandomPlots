import os
import sys
import cv2
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt


def vectorized_sample_complex_pairs(sample_size: int):
    # Sample 2n random angles from 0 to 2*pi (2 for each pair)
    np.random.seed(1)
    thetas = np.random.uniform(0, 2 * np.pi, size=2 * sample_size)

    # Compute complex numbers
    zs = np.exp(1j * thetas)

    # Reshape to get n pairs
    pairs = zs.reshape(sample_size, 2)

    return pairs


def calculate_matrix(t):
    return np.array([[-1j, 0, -1j, 0.5, -1j],
                     [-1j, 1, -1j, 0, 0],
                     [0, t[1], -1j, 0.5, 1],
                     [1, -1j, 1j, 0.5, 1j],
                     [0, 1j, t[0], 0, 1]])
    # return np.array([[t[0], 1j],
    #                  [-0.5, t[1]]])


def calculate_eigenvalues(x: np.array):
    return np.linalg.eigvals(x)


def generate_plot(x, y, directory):
    current_time = datetime.now()
    time_string = current_time.strftime('%Y-%m-%d_%H-%M-%S-%f')
    fig, _ = plt.subplots(figsize=(12, 12), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1], facecolor='#f4f0e7')
    ax.scatter(x, y, color='k', s=5, lw=0)
    ax.set_xlim(-2, 3)
    ax.set_ylim(-3, 2)
    if not os.path.exists(f"outputs/{directory}"):
        os.makedirs(f"outputs/{directory}")
    fig.savefig(f'outputs/{directory}/{time_string}.png', facecolor='k')
    plt.close()


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
    out = cv2.VideoWriter("outputs/" + video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for image in images:
        img_path = os.path.join(image_folder, image)
        img = cv2.imread(img_path)
        out.write(img)

    out.release()


def generate():
    filename = sys.argv[1]
    sample_size = 5000
    sample = vectorized_sample_complex_pairs(sample_size)
    for i, r in enumerate(np.linspace(0, 6, 600)):
        if i> 365:
            Z = np.array([calculate_eigenvalues(calculate_matrix(t)) for t in sample * r]).ravel()
            generate_plot(Z.real, Z.imag, filename)
    images_to_video(f"outputs/{filename}", f'{filename}.mp4', 20)
