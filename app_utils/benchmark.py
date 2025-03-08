import csv
import importlib
import os
import timeit
from datetime import datetime

import numpy as np
from loguru import logger

# Directory containing the figures modules
figures_dir = "figures"

# Path to the CSV file for results
results_file = "benchmark_results.csv"

# Define headers
headers = ["timestamp", "script", "n_runs", "average", "q99", "min", "max"]

# List all scripts in the figures directory
scripts = [f[:-3] for f in os.listdir(figures_dir) if f.endswith(".py") and f != "__init__.py"]

# Check if the results file already exists
file_exists = os.path.isfile(results_file)

# Write the header if the file does not exist
if not file_exists:
    with open(results_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
    logger.info(f"Created new benchmark results file: {results_file}")

# Start benchmarking
for script in scripts:
    if script != 'cubism':
        continue
    try:
        module = importlib.import_module(f"figures.{script}")
        logger.info(f"Benchmarking script: {script}")
        n_runs = 20

        # Check if the create_image function exists
        if not hasattr(module, "create_image"):
            logger.error(f"Script {script} does not have a 'create_image' function. Skipping.")
            continue

        # Warm-up
        module.create_image()

        # Measure timing
        times = [timeit.timeit(lambda: module.create_image(), number=1) for _ in range(n_runs)]
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
