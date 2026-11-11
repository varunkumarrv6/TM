#!/usr/bin/env bash
# Build, run all 10 kernels on the Hamming-encoded datagram stream, check correctness, write results.csv.
#   ./run_all.sh [data dir=data] [seconds of timed inference per kernel=1.0] [stream=medsec_tm_hamming.bin]
set -u
cd "$(dirname "$0")"
DATA="${1:-data}"; BUDGET="${2:-1.0}"; STREAM="${3:-medsec_tm_hamming.bin}"
make -s all || exit 1
echo "== host: $(uname -m), $(nproc) cpus, $(grep -m1 -E 'model name|Model' /proc/cpuinfo | cut -d: -f2)"
out=results.csv
echo "kernel,threads,stream,layout,receive_ms,corrected,prep_ms,infer_ms_per_sample,batch_ms,reps,hash_ok,scores_ok,tmu_match,accuracy,all_ok,detect_ms,correct_ms,e2e_ms" > "$out"
fail=0
for k in tm01_normal tm02_packed_bytes tm03_packed_early_exit tm04_packed64 tm05_packed_batch \
         tm06_scalar_neon tm07_persistent_neon tm08_interclause_neon tm09_interclause_omp tm10_final_neon; do
    res="$(./bin/$k "$DATA" "$BUDGET" "$STREAM" 2>/dev/null)"; rc=$?
    echo "$res" | grep -v '^RESULT|'
    [ $rc -ne 0 ] && { echo "   ^^^ FAILED (exit $rc)"; fail=1; }
    echo "$res" | grep '^RESULT|' | cut -d'|' -f2- | tr '|' ',' >> "$out"
done
echo
printf "%-32s %10s %10s %14s %12s %10s\n" "kernel" "decode ms" "ECC ms" "infer ms/smp" "e2e ms" "speed-up"
base=$(awk -F, 'NR == 2 {print $8}' "$out")
awk -F, -v b="$base" 'NR > 1 {printf "%-32s %10.3f %10.3f %14.5f %12.3f %9.1fx  %s\n", $1 " (" $2 "T)", $16, $17, $8, $18, b / $8, ($15 == 1 ? "OK" : "CHECK FAILED")}' "$out"
echo "(per-kernel results: $out)"
exit $fail
