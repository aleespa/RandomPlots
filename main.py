import argparse
import importlib
import sys

from loguru import logger

from common.image_processing import ImageProcessingSettings


def load_script_function(module_path):
    """
    Dynamically import the module and return its generate function.
    """
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        logger.exception(f"Failed to import module '{module_path}'")
        return None
    func = getattr(module, "generate", None)
    if func is None:
        logger.error(f"No 'generate()' function found in module '{module_path}'")
    return func


def output_name(module_path):
    """
    Turn a module path into an output folder name, dropping the top-level package:
    'lab.2026.0314' -> '20260314', 'static.polygons' -> 'polygons'.
    """
    names = module_path.split(".")
    return "".join(names[1:]) or names[0]


@logger.catch
def main():
    parser = argparse.ArgumentParser(description="Run a specific plot generator.")
    parser.add_argument(
        "script",
        type=str,
        help="Module path to run (e.g., 'animations.rain' or 'lab.2026.0314')",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for the generator's RNG"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Output folder name under outputs/ (defaults to the module path)",
    )
    args = parser.parse_args()

    logger.info(f"Running script: {args.script}")
    func = load_script_function(args.script)
    if func is None:
        return 1

    settings = ImageProcessingSettings(
        seed=args.seed, filename=args.name or output_name(args.script)
    )
    try:
        func(settings)
    except Exception:
        logger.exception(f"Error while executing script '{args.script}'")
        return 1

    logger.success(f"Finished: {settings.output_path / settings.filename}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
