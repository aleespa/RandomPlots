import base64
import importlib.util
import os
import traceback

import numpy as np
from loguru import logger

# Paths for scripts and outputs
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
RUNS_PER_SCRIPT = 100

# Ensure the output directory exists
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def load_create_image_function(script_path):
    """Dynamically load the `create_image` function from a script."""
    script_name = os.path.splitext(os.path.basename(script_path))[0]
    spec = importlib.util.spec_from_file_location(script_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_image


def save_image(base64_str, output_path):
    """Decode the Base64 string and save it as a binary image file."""
    try:
        image_data = base64.b64decode(base64_str)
        with open(output_path, "wb") as f:
            f.write(image_data)
    except Exception as e:
        logger.info(f"Error saving image to {output_path}: {e}")


# Iterate over all Python scripts in the figures directory
for script_file in os.listdir(FIGURES_DIR):

    if script_file != 'bw.py':
        continue
    script_path = os.path.join(FIGURES_DIR, script_file)

    # Skip non-Python files
    if not script_file.endswith(".py"):
        continue

    # Load the `create_image` function
    try:
        create_image = load_create_image_function(script_path)
    except AttributeError:
        logger.info(f"Script {script_file} does not contain a `create_image` function. Skipping.")
        continue

    # Create a directory for the script's outputs
    script_output_dir = os.path.join(OUTPUTS_DIR, os.path.splitext(script_file)[0])
    os.makedirs(script_output_dir, exist_ok=True)

    # Run the function 100 times and save the outputs
    for i in range(RUNS_PER_SCRIPT):
        try:
            base64_str = create_image(np.random.randint(0, 1000000), True, (0, 13/255, 30/255))
            if not isinstance(base64_str, str):
                logger.info(f"Error: `create_image` in {script_file} did not return a Base64 string. Skipping this run.")
                continue

            output_path = os.path.join(script_output_dir, f"image_{i + 1:03d}.jpg")
            save_image(base64_str, output_path)
        except Exception as e:
            logger.info(f"Error running `create_image` in {script_file}: {e}")
            continue

    for i in range(RUNS_PER_SCRIPT):
        try:
            base64_str = create_image(np.random.randint(0, 1000000), False, (244/255, 240/255, 231/255))
            if not isinstance(base64_str, str):
                logger.info(
                    f"Error: `create_image` in {script_file} did not return a Base64 string. Skipping this run.")
                continue

            output_path = os.path.join(script_output_dir, f"image_light_{i + 1:03d}.jpg")
            save_image(base64_str, output_path)
        except Exception as e:
            tb = traceback.format_exc()
            logger.info(f"Error running `create_image` in {script_file}\n{tb}")
            continue

    logger.info(f"Finished processing {script_file}")
