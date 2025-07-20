import argparse
import importlib

from loguru import logger


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

    logger.info(f"Attempting to run script: {args.script}")
    func = load_script_function(args.script)
    if func:
        try:
            func(*args.args)
        except Exception as e:
            logger.exception(f"Error while executing script '{args.script}'")


if __name__ == "__main__":
    main()
