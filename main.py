import importlib

import toml
from loguru import logger

logger.add("app_log.log", rotation="10 MB", level="INFO")


def run_script(y, m):
    module_path = f"lab.{y}.{m}"
    script_module = importlib.import_module(module_path)
    script_module.generate()


if __name__ == "__main__":
    config = toml.load('config.toml')
    file_to_run = config['file_to_run']

    year = file_to_run[:4]
    month_day = file_to_run[4:]

    logger.info(f"Running script: {year}{month_day}")
    run_script(year, month_day)
