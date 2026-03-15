import gc
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.random import Generator

OUTPUT_PATH = Path(__file__).parent.parent / 'outputs'


class ImageProcessingSettings:
    filename: str
    output_path: Path = OUTPUT_PATH
    rng: Generator

    def __init__(self, seed=0, filename=None):
        self.filename = filename or datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        self.rng = np.random.default_rng(seed=seed)
        create_directory(self.output_path / self.filename)
        clear_folder(self.output_path / self.filename)

    def save_to_png(self, fig:plt.Figure, bg_color='#000000'):
        time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        fig.savefig(
            self.output_path / self.filename / f'{time_string}.png', facecolor=bg_color
        )

    def save_frame(self, face_color='#000000'):
        time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        plt.savefig(
            self.output_path / self.filename / f'{time_string}.png',
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
        images = [
            img
            for img in os.listdir(OUTPUT_PATH / self.filename)
            if img.endswith(".PNG") or img.endswith(".png")
        ]
        images.sort()  # Make sure the images are in the correct order

        # Get the size of the images
        frame = cv2.imread(str(OUTPUT_PATH / self.filename / images[0]))
        h, w, layers = frame.shape
        size = (w, h)

        # Create a video writer object
        out = cv2.VideoWriter(
            (OUTPUT_PATH / (self.filename + self.filename + '.mp4')).__str__(),
            cv2.VideoWriter_fourcc(*'avc1'),
            fps,
            size,
        )

        for image in images:
            img_path = str(self.output_path / self.filename / image)
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
