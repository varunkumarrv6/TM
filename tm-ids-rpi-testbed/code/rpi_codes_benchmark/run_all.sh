#!/usr/bin/env bash
# Build, run every kernel, check correctness, print a summary table.
#   ./run_all.sh [iot_dataset dir] [seconds per kernel]
set -u
cd "$(dirname "$0")"
DATA="$(cd "${1:-..}" && pwd)"
BUDGET="${2:-1.0}"

make -s all || exit 1

echo "== host: $(uname -m), $(nproc) cpus, $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 || true)"
echo

results=()
fail=0
for k in tm01_normal tm02_packed_bytes tm03_packed_early_exit tm04_packed64 \
         tm05_packed_batch tm06_scalar_neon tm07_persistent_neon \
         tm08_interclause_neon tm09_interclause_omp tm10_final_neon; do
    out="$(./bin/$k "$DATA" "$BUDGET")"; rc=$?
    echo "$out" | grep -v '^RESULT|'
    [ $rc -ne 0 ] && { echo "   ^^^ FAILED (exit $rc)"; fail=1; }
    results+=("$(echo "$out" | grep '^RESULT|')")
done

echo
echo "== original/tm_interclause_neon_single.c (unmodified, sample 0)"
mkdir -p original_run
( cd original_run && ../bin/export_bins "$DATA" 0 >/dev/null && ../bin/original_interclause ) \
    | grep -E "Scores|Reference|1-core avg" || fail=1

echo
base=$(echo "${results[0]}" | cut -d'|' -f3)
printf "%-32s %14s %10s %11s %9s\n" "kernel" "ms/sample" "speedup" "scores=ref" "TMU"
for r in "${results[@]}"; do
    IFS='|' read -r _ name ms batch reps ok tmu acc <<< "$r"
    printf "%-32s %14.4f %9.1fx %11s %6s/50\n" "$name" "$ms" \
        "$(awk -v a="$base" -v b="$ms" 'BEGIN{print a/b}')" \
        "$([ "$ok" = 1 ] && echo yes || echo NO)" "$tmu"
done
exit $fail
