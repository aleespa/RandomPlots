import gc
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import toml
from matplotlib import pyplot as plt
from numpy.random import Generator

OUTPUT_PATH = Path('outputs')


class ImageProcessingSettings:
    filename: str
    rng: Generator

    def __init__(self, seed=0):
        try:
            config = toml.load('config.toml')
        except FileNotFoundError:
            print("Error: config.toml not found!")
            raise
        except toml.TomlDecodeError:
            print("Error: Failed to parse config.toml!")
            raise
        self.filename = config['file_to_run']
        self.rng = np.random.default_rng(seed=seed)
        create_directory(OUTPUT_PATH / self.filename)
        clear_folder(OUTPUT_PATH / self.filename)

    def save_to_png(self, face_color='#000000'):
        plt.savefig(
            OUTPUT_PATH / self.filename / 'figure.png',
            facecolor=face_color
        )

    def save_frame(self, face_color='#000000'):
        time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        plt.savefig(
            OUTPUT_PATH / self.filename / f'{time_string}.png',
            facecolor=face_color,
        )
        plt.close()
        gc.collect()

    def images_to_video(self, fps=30):
        """
        Convert a folder of images into an MP4 video.

        Parameters:
        - image_folder: Folder containing image frames.
        - video_name: Name of the output video file (including .mp4 extension).
        - fps: Frames per second for the output video.
        """
        images = [img for img in os.listdir(f'outputs/{self.filename}')
                  if img.endswith(".PNG") or img.endswith(".png")]
        images.sort()  # Make sure the images are in the correct order

        # Get the size of the images
        frame = cv2.imread(os.path.join(f'outputs/{self.filename}', images[0]))
        h, w, layers = frame.shape
        size = (w, h)

        # Create a video writer object
        out = cv2.VideoWriter((OUTPUT_PATH
                              / (self.filename + self.filename + '.mp4')).__str__(),
                              cv2.VideoWriter_fourcc(*'avc1'),
                              fps, size)

        for image in images:
            img_path = os.path.join(OUTPUT_PATH / self.filename, image)
            img = cv2.imread(img_path)
            out.write(img)

        out.release()


def clear_folder(folder_path):
    # List all files and directories inside the folder
    for file_name in os.listdir(folder_path):
        # Construct the full path for each file/directory
        file_path = os.path.join(folder_path, file_name)

        # Check if it's a file
        if os.path.isfile(file_path):
            # If it's a file, delete it
            os.remove(file_path)
        elif os.path.isdir(file_path):
            # If it's a directory, call the function recursively to clear it
            clear_folder(file_path)
            # After clearing the subdirectory, remove the directory itself
            os.rmdir(file_path)


def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
