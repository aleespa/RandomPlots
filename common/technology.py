import os
import shutil
import subprocess
from pathlib import Path

from loguru import logger


def has_nvidia_gpu():
    """
    Check if an NVIDIA GPU is available by looking for 'nvidia-smi' executable.
    """
    return shutil.which("nvidia-smi") is not None

def images_to_video(image_folder: Path, video_name, fps):
    """
    Convert a folder of images into a lossless MP4 video using FFmpeg.

    Parameters:
    - image_folder: Folder containing image frames.
    - video_name: Name of the output video file (including .mp4 extension).
    - fps: Frames per second for the output video.
    """
    # Choose codec depending on GPU availability
    if has_nvidia_gpu():
        logger.info("NVIDIA GPU detected — using h264_nvenc lossless.")
        codec = 'h264_nvenc'
        codec_options = ['-qp', '0', '-preset', 'slow']
    else:
        logger.warning("No NVIDIA GPU found — using CPU (libx264 lossless).")
        codec = 'libx264'
        codec_options = ['-crf', '0', '-preset', 'veryslow']

    # Sort images
    images = [img for img in os.listdir(image_folder) if img.lower().endswith(".png")]
    images.sort()

    if not images:
        logger.warning("No PNG images found in folder.")
        return

    first_img_name = images[0]
    name_parts = os.path.splitext(first_img_name)
    if not any(c.isdigit() for c in name_parts[0]):
        logger.warning(
            "Warning: Your images do not contain numbers in their names. "
            "FFmpeg expects numbered patterns (e.g., frame_0001.png)."
        )

    # Output paths
    output_folder = image_folder.parent
    output_path = output_folder / video_name

    images_path = image_folder / 'frame%04d.png'

    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-framerate', str(fps),
        '-i', images_path.absolute().__str__(),
        '-c:v', codec,
        *codec_options,
        '-pix_fmt', 'yuv444p',   # Use 4:4:4 for true lossless color if supported
        output_path.absolute().__str__(),
    ]
    result = subprocess.run(
        ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        logger.error(f"FFmpeg error:\n{result.stderr}")

    logger.success(f"Video saved to {output_path}")

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
