import argparse
import importlib

from loguru import logger

# Configure Loguru to log both to console and file
logger.add("app_log.log", rotation="10 MB", level="INFO")

def run_script(y, m):
    module_path = f"lab.{y}.{m}"
    script_module = importlib.import_module(module_path)
    script_module.generate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the script for a specific date.')
    parser.add_argument('date', type=str, help='Date in the format YYYYMMDD')

    args = parser.parse_args()

    year = args.date[:4]
    month_day = args.date[4:]

    run_script(year, month_day)
