import csv
import importlib
import os
import timeit
from datetime import datetime

import matplotlib.colors as mcolors
import numpy as np
from loguru import logger


def main():
    figures_dir = "figures"
    results_file = "benchmark_results.csv"
    headers = ["timestamp", "script", "n_runs", "average", "q99", "min", "max"]
    scripts = [f[:-3] for f in os.listdir(figures_dir) if f.endswith(".py") and f != "__init__.py"]
    file_exists = os.path.isfile(results_file)
    if not file_exists:
        with open(results_file, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
        logger.info(f"Created new benchmark results file: {results_file}")

    cmap = create_colormap(["#000000", "#335325", "#ffffff"])
    for script in scripts:
        try:
            module = importlib.import_module(f"figures.{script}")
            logger.info(f"Benchmarking script: {script}")
            n_runs = 100

            # Check if the create_image function exists
            if not hasattr(module, "create_image"):
                logger.error(f"Script {script} does not have a 'create_image' function. Skipping.")
                continue

            module.create_image(cmap=cmap)

            times = [timeit.timeit(lambda: module.create_image(cmap=cmap), number=1) for _ in range(n_runs)]
            metrics = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "script": script,
                "n_runs": n_runs,
                "average": np.mean(times),
                "q99": np.quantile(times, 0.99),
                "min": np.min(times),
                "max": np.max(times),
            }

            # Append the result for this script to the CSV file
            with open(results_file, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writerow(metrics)

            logger.info(f"Results for script {script} appended to {results_file}")

        except Exception as e:
            logger.error(f"Error in script {script}: {e}")

    logger.info("Benchmarking completed.")


def create_colormap(color_list):
    n = len(color_list)
    stops = [(i / (n - 1), c) for i, c in enumerate(color_list)]
    return mcolors.LinearSegmentedColormap.from_list("custom_map", stops)

if __name__ == "__main__":
    main()
