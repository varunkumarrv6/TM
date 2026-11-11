#!/usr/bin/env bash
# Full pipeline: preprocess -> train 3 TM nodes -> attack/ECC simulation -> figures
set -e
cd "$(dirname "$0")"
PY=/home/<user>/iot_dataset/proj/bin/python3   # the 'proj' venv that has tmu + numba
"$PY" prep.py
for n in 0 1 2; do "$PY" train_node.py $n 15 1000 1000 5.0 2>/dev/null > train_node$n.log & done; wait
"$PY" run_attacks.py 2>/dev/null
"$PY" plots.py
