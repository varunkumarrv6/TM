#!/usr/bin/env bash
# Single core vs OpenMP threads: kernels 08 and 10 (1 core) against kernel 09 with TM_THREADS = 1..4,
# REPS runs each, on the Hamming stream.  Writes threads_results.csv (one row per run) and prints medians.
#   ./run_threads.sh [data dir=data] [seconds per run=1.0] [reps=5]
set -u
cd "$(dirname "$0")"
DATA="${1:-data}"; BUDGET="${2:-1.0}"; REPS="${3:-5}"
make -s all || exit 1
out=threads_results.csv
echo "kernel,threads,rep,receive_ms,infer_ms_per_sample,all_ok" > "$out"
run() {   # $1 = binary, $2 = threads
    for r in $(seq 1 "$REPS"); do
        line="$(TM_THREADS="$2" ./bin/"$1" "$DATA" "$BUDGET" 2>/dev/null | grep '^RESULT|')"
        IFS='|' read -r _ name thr stream lay rms corr prep ms batch reps hok sok tmu acc ok det cor e2e <<< "$line"
        echo "$1,$2,$r,$rms,$ms,$ok" >> "$out"
        printf "%-24s %sT  rep %d  receive %.3f ms  infer %.5f ms/sample\n" "$1" "$2" "$r" "$rms" "$ms"
    done
}
run tm08_interclause_neon 1
run tm10_final_neon 1
for t in 1 2 3 4; do run tm09_interclause_omp "$t"; done
echo
echo "== median (lower is better), host: $(uname -m), $(nproc) cpus"
printf "%-30s %12s %14s\n" "kernel" "receive ms" "infer ms/smp"
for col in 4 5; do
    tail -n +2 "$out" | sort -t, -k1,1 -k2,2n -k${col},${col}g |
        awk -F, -v c=$col '{ k = $1 " " $2 "T"; a[k, ++n[k]] = $c } END { for (k in n) print k "," a[k, int((n[k] + 1) / 2)] }' |
        sort > ".med$col"
done
join -t, .med4 .med5 | awk -F, '{printf "%-30s %12.3f %14.5f\n", $1, $2, $3}'
rm -f .med4 .med5
echo "(per-run results: $out)"
