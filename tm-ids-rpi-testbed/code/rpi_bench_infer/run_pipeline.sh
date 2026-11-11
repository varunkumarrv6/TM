#!/usr/bin/env bash
# End-to-end latency (first packet available -> class sums of all test samples) for the 4 receive modes of
# tm11_pipeline: seq / stream / swp (1 core) and pipe (3 cores), on the Hamming, error and raw streams.
#   ./run_pipeline.sh [data dir=data] [seconds per mode=2.0]   -> pipeline_results.csv
set -u
cd "$(dirname "$0")"
DATA="${1:-data}"; BUDGET="${2:-2.0}"
make -s all || exit 1
out=pipeline_results.csv
echo "mode,cores,stream,layout,e2e_ms,speedup_vs_seq,reps,corrected,scores_ok,all_ok,decode_busy_ms,ecc_busy_ms,infer_busy_ms" > "$out"
fail=0
for s in medsec_tm_hamming.bin medsec_tm_hamming_err.bin medsec_tm_raw.bin; do
    res="$(./bin/tm11_pipeline "$DATA" "$BUDGET" "$s")" || fail=1
    echo "$res" | grep -v '^RESULT|'
    echo "$res" | grep '^RESULT|' | cut -d'|' -f2- | tr '|' ',' >> "$out"
done
echo "(per-mode results: $out)"
exit $fail
