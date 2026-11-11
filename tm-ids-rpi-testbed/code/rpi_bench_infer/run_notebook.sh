#!/usr/bin/env bash
# Runs benchmark_rpi.ipynb top to bottom (Python -> NumPy -> C kernels -> threads -> pipelining) and stores the
# executed notebook plus results_rpi/ (figures PDF + PNG, all_results.csv, summary.txt).
# Without Jupyter it runs the same code from benchmark_rpi_cells.py.
set -u
cd "$(dirname "$0")"
if command -v jupyter >/dev/null 2>&1; then
    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 benchmark_rpi.ipynb
else
    echo "jupyter not found -> running benchmark_rpi_cells.py with python3"
    python3 "$(pwd)/benchmark_rpi_cells.py"
fi
echo; cat results_rpi/summary.txt
