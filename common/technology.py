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

def images_to_video(image_folder: Path, video_name, fps, crf: int = 0):
    """
    Convert a folder of images into an MP4 video using FFmpeg.

    Parameters:
    - image_folder: Folder containing image frames.
    - video_name: Name of the output video file (including .mp4 extension).
    - fps: Frames per second for the output video.
    - crf: 0 (default) encodes lossless -- large files, meant for archival
      masters. Any value > 0 switches to constant-quality encoding at that
      quality level (lower = better/bigger; ~18-23 is visually near-lossless
      but a fraction of the size -- appropriate for uploading to Instagram,
      which recompresses on ingest anyway).
    """
    # Choose codec depending on GPU availability
    if has_nvidia_gpu():
        codec = 'h264_nvenc'
        if crf == 0:
            logger.info("NVIDIA GPU detected — using h264_nvenc lossless.")
            codec_options = ['-qp', '0', '-preset', 'slow']
        else:
            logger.info(f"NVIDIA GPU detected — using h264_nvenc, cq={crf}.")
            codec_options = ['-rc', 'vbr', '-cq', str(crf), '-b:v', '0', '-preset', 'p5']
    else:
        codec = 'libx264'
        if crf == 0:
            logger.warning("No NVIDIA GPU found — using CPU (libx264 lossless).")
            codec_options = ['-crf', '0', '-preset', 'veryslow']
        else:
            logger.warning(f"No NVIDIA GPU found — using CPU (libx264, crf={crf}).")
            codec_options = ['-crf', str(crf), '-preset', 'medium']

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
        '-pix_fmt', 'yuv420p',   # H.264 standard format, widely supported on Windows
        output_path.absolute().__str__(),
    ]
    result = subprocess.run(
        ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        logger.error(f"FFmpeg error:\n{result.stderr}")

    logger.success(f"Video saved to {output_path}")


def images_to_hdr_video(
    image_folder: Path,
    video_name: str,
    fps: int,
    width: int = 1080,
    height: int = 1920,
    bitrate: str = "24M",
):
    """Encode numbered PNG frames as an Instagram-oriented HLG HDR10 video.

    The PNG frames are assumed to be sRGB/BT.709 artwork. FFmpeg converts them
    to BT.2020 HLG, then encodes 10-bit 4:2:0 HEVC. This produces a valid HDR
    delivery file; genuine additional highlight latitude requires an HDR-aware
    renderer upstream of this function.
    """
    images = sorted(
        img for img in os.listdir(image_folder) if img.lower().endswith(".png")
    )
    if not images:
        logger.warning("No PNG images found in folder.")
        return

    output_path = image_folder.parent / video_name
    images_path = image_folder / "frame%04d.png"

    # zscale performs the colour conversion from BT.709/sRGB to BT.2020 HLG.
    # The explicit format conversion guarantees a 10-bit HEVC-compatible input.
    vf = (
        f"scale={width}:{height}:flags=lanczos:force_original_aspect_ratio=disable,"
        "zscale=matrixin=bt709:primariesin=bt709:transferin=bt709:"
        "matrix=bt2020nc:primaries=bt2020:transfer=arib-std-b67:npl=1000:range=limited,"
        "format=yuv420p10le"
    )
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        images_path.absolute().__str__(),
        "-vf",
        vf,
        "-c:v",
        "libx265",
        "-preset",
        "medium",
        "-b:v",
        bitrate,
        "-maxrate",
        bitrate,
        "-bufsize",
        "48M",
        "-pix_fmt",
        "yuv420p10le",
        "-profile:v",
        "main10",
        "-tag:v",
        "hvc1",
        "-color_primaries",
        "bt2020",
        "-colorspace",
        "bt2020nc",
        "-color_trc",
        "arib-std-b67",
        "-color_range",
        "tv",
        "-movflags",
        "+faststart",
        output_path.absolute().__str__(),
    ]
    result = subprocess.run(
        ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        logger.error(f"FFmpeg HDR error:\n{result.stderr}")
        return

    logger.success(f"HDR video saved to {output_path}")

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
