# MedSec-25: MITM attacks on a distributed weighted Tsetlin Machine IDS + Hamming(7,4)

Dataset: MedSec-25 (Kaggle `abdullah001234/medsec-25-iomt-cybersecurity-dataset`, CC BY-SA 4.0) in `raw/`.

| File | Step |
|---|---|
| `prep.py` | clean, dedupe, 80/20 stratified split, quantile-thermometer booleanization (178 bits -> 356 literals), 3 disjoint node shards -> `data.npz` |
| `tune.py` | hyperparameter sweep on node-0's own validation split |
| `train_node.py N` | train weighted TM node N (tmu, Cl=1000/class, T=1000, s=5), export include/exclude cube + weights -> `nodeN.npz` |
| `sim.py` | numba inference engine (verified bit-exact vs tmu), fusion of clipped class sums, attacks A1/A2/A2+/A3/A4, Hamming(7,4) with SEQ/INT/KEY grouping -> `results.csv` |
| `plots.py` | figures `fig_*.png` and tables `table_*node.csv` |
| `run_all.sh` | runs everything end to end |

## File-level ECC + TM inference (same flow as `error_coding.ipynb` cells 53-55)
| File | Step |
|---|---|
| `ecc_pipeline.py` | sender (clause banks -> packets -> Hamming(7,4) -> `tx/tm{n}_{layout}.bin`), MITM attacks on the file bytes, model-blind decoder, manual TM inference |
| `ecc_tm_inference.ipynb` | step-by-step run: encode -> corrupt -> decode -> rule check -> manual inference vs tmu original -> sweep of all attacks x layouts (`ecc_pipeline_summary.csv`) |
| `medsciot_results.ipynb` | figures from the in-memory simulation (`sim.py`) |
