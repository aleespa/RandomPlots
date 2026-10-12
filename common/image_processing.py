from __future__ import annotations

import gc
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.random import Generator

from common.technology import images_to_video

if TYPE_CHECKING:
    from matplotlib.figure import Figure

OUTPUT_PATH = Path(__file__).parent.parent / 'outputs'


class ImageProcessingSettings:
    filename: str
    output_path: Path = OUTPUT_PATH
    rng: Generator

    def __init__(
        self, seed=0, filename=None, hdr=False, audio_only=None, audio_seed=None
    ):
        """
        `audio_only`: None for a normal run, which clears the output folder. A
        string (possibly empty) re-runs only the soundtrack over the frames of
        an earlier full run in the same folder, which is kept: `render_frames`
        skips, `save_video` reuses the silent master, and a non-empty string is
        added to the output names (`<name>_<tag>.mp4`) so auditions coexist.

        `audio_rng` is the soundtrack's own generator, seeded by `audio_seed`
        (default: `seed`), so the music can be re-rolled without touching the
        picture.
        """
        self.filename = filename or datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self.audio_seed = seed if audio_seed is None else audio_seed
        self.audio_rng = np.random.default_rng([self.audio_seed, 1])
        self.hdr = hdr
        self.audio_only = audio_only is not None
        self.audio_tag = audio_only or ''
        folder = self.output_path / self.filename
        if self.audio_only:
            if not (folder / 'frames').is_dir():
                raise FileNotFoundError(
                    f'--audio-only needs the frames of a full run in {folder}: '
                    'render the video first, with the same --name'
                )
            return
        create_directory(folder)
        clear_folder(folder)

    def save_to_png(self, fig: Figure, bg_color='#000000'):
        time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        fig.savefig(
            self.output_path / self.filename / f'{time_string}.png', facecolor=bg_color
        )

    def save_frame(self, face_color='#000000'):
        from matplotlib import pyplot as plt

        time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        plt.savefig(
            self.output_path / self.filename / f'{time_string}.png',
            facecolor=face_color,
        )
        plt.close()
        gc.collect()

    @property
    def frames_path(self) -> Path:
        """Folder holding the numbered frames of an animation, created on first use."""
        path = self.output_path / self.filename / 'frames'
        create_directory(path)
        return path

    def save_numbered_frame(self, index: int, face_color='#000000'):
        """
        Save the current figure as frames/frame%04d.png -- the naming ffmpeg expects,
        and what `save_video` reads back.
        """
        from matplotlib import pyplot as plt

        plt.savefig(self.frames_path / f'frame{index:04d}.png', facecolor=face_color)
        plt.close()
        gc.collect()

    def save_video(self, fps=30, crf=0, audio=None, sample_rate=None):
        """
        Encode the frames folder into outputs/<filename>/<filename>.mp4 with ffmpeg.

        `crf=0` (default) is lossless. Pass a higher crf (e.g. 18-23) for a much
        smaller, still visually near-lossless file -- appropriate for uploading
        to Instagram rather than archiving. Ignored when `hdr=True`.

        `audio` is an optional soundtrack, a float array of shape (samples, 2)
        (or mono (samples,)) in [-1, 1], as long as the video -- see
        `common.audio`. The silent encode is kept as <video>_silent.mp4, the
        track is saved as <filename>.wav plus a <filename>_spectrogram.png to
        look at, and the two are muxed (AAC) into the final video. An
        audio-only run reuses the silent master, so only the mux happens, and
        its tag (if any) is added to every output name.
        """
        folder = self.output_path / self.filename
        video_name = f'{self.filename}_HDR_HLG.mp4' if self.hdr else f'{self.filename}.mp4'
        silent = folder / video_name.replace('.mp4', '_silent.mp4')
        if self.audio_only and audio is None:
            raise ValueError('--audio-only, but generate() passed no soundtrack')

        if not (self.audio_only and silent.exists()):
            if self.hdr:
                from common.technology import images_to_hdr_video

                images_to_hdr_video(self.frames_path, video_name, fps)
            else:
                images_to_video(self.frames_path, video_name, fps, crf=crf)
            if audio is not None:
                os.replace(folder / video_name, silent)
        if audio is None:
            return

        from common.audio import SAMPLE_RATE, spectrogram_png, write_wav
        from common.technology import add_audio_to_video

        sr = sample_rate or SAMPLE_RATE
        suffix = f'_{self.audio_tag}' if self.audio_tag else ''
        wav_path = folder / f'{self.filename}{suffix}.wav'
        write_wav(wav_path, audio, sr)
        spectrogram_png(folder / f'{self.filename}{suffix}_spectrogram.png', audio, sr)
        output = folder / video_name.replace('.mp4', f'{suffix}.mp4')
        add_audio_to_video(silent, wav_path, output)

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
