import argparse
import importlib


def run_script(y, m):
    # Construct the module path dynamically
    module_path = f"labs.{y}.{m}"

    # Dynamically import the module
    script_module = importlib.import_module(module_path)

    # Assume each script has a function named `generate_plot`
    script_module.generate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the script for a specific date.')
    parser.add_argument('date', type=str, help='Date in the format YYYYMMDD')

    args = parser.parse_args()

    year = args.date[:4]
    month_day = args.date[4:]

    run_script(year, month_day)
