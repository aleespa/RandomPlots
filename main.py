import argparse
import importlib

from loguru import logger

from common.image_processing import ImageProcessingSettings


def load_script_function(module_path):
    """
    Dynamically import the module and return its main or run function.
    """
    try:
        module = importlib.import_module(module_path)
        if hasattr(module, 'generate'):
            return module.generate
        else:
            logger.error(f"No 'generate()' function found in module '{module_path}'")
            return None
    except ModuleNotFoundError as e:
        logger.exception(f"Failed to import module '{module_path}'")
        return None

@logger.catch
def main():
    parser = argparse.ArgumentParser(description="Run a specific plot generator.")
    parser.add_argument(
        "script",
        type=str,
        help="Script path to run (e.g., 'animations.wave' or 'static.circle')"
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed to the script"
    )
    args = parser.parse_args()

    logger.info(f"Running script: {args.script}")
    func = load_script_function(args.script)
    settings = ImageProcessingSettings()
    names = args.script.split('.')
    settings.filename = names[1]+names[2]
    if func:
        try:
            func(settings)
        except Exception as e:
            logger.exception(f"Error while executing script '{args.script}'")


if __name__ == "__main__":
    main()
