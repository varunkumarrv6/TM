#!/usr/bin/env bash
# Single thread (1T) vs OpenMP threads on the Raspberry Pi: why the deployed TM uses one core.
# Runs the single-thread kernels (08 inter-clause NEON, 10 final NEON) and the OpenMP kernel (09) with
# TM_THREADS = 1, 2, 3, 4, REPS times each, and writes threads_results.csv (one row per run).
#   ./run_threads.sh [data dir=..] [seconds per run=1.0] [reps=5]
set -u
cd "$(dirname "$0")"
DATA="$(cd "${1:-..}" && pwd)"
BUDGET="${2:-1.0}"
REPS="${3:-5}"
make -s all || exit 1

out=threads_results.csv
echo "kernel,threads,rep,ms_per_sample,scores_match,tmu_agree" > "$out"
run() {   # $1 = binary, $2 = threads label
    for r in $(seq 1 "$REPS"); do
        line="$(TM_THREADS="${2}" ./bin/"$1" "$DATA" "$BUDGET" | grep '^RESULT|')"
        IFS='|' read -r _ name ms batch reps ok tmu acc <<< "$line"
        echo "$1,$2,$r,$ms,$ok,$tmu" >> "$out"
        printf "%-26s %sT  rep %d  %.4f ms/sample\n" "$1" "$2" "$r" "$ms"
    done
}
run tm08_interclause_neon 1
run tm10_final_neon 1
for t in 1 2 3 4; do run tm09_interclause_omp "$t"; done

echo
echo "== median ms/sample (lower is better), host: $(uname -m), $(nproc) cpus"
tail -n +2 "$out" | sort -t, -k1,1 -k2,2n -k4,4g |          # portable median (works with mawk on the Pi)
    awk -F, '{ k = $1 " " $2 "T"; a[k, ++n[k]] = $4 }
             END { for (k in n) printf "%-30s %.4f\n", k, a[k, int((n[k] + 1) / 2)] }' | sort
echo "(results per run: $out)"
