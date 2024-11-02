import os

import cv2


def images_to_video(image_folder, video_name, fps):
    """
    Convert a folder of images into an MP4 video.

    Parameters:
    - image_folder: Folder containing image frames.
    - video_name: Name of the output video file (including .mp4 extension).
    - fps: Frames per second for the output video.
    """
    images = [img for img in os.listdir(image_folder)
              if img.endswith(".PNG") or img.endswith(".png")]
    images.sort()  # Make sure the images are in the correct order

    # Get the size of the images
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    h, w, layers = frame.shape
    size = (w, h)

    # Create a video writer object
    out = cv2.VideoWriter("outputs/" + video_name,
                          cv2.VideoWriter_fourcc(*'avc1'),
                          fps, size)

    for image in images:
        img_path = os.path.join(image_folder, image)
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