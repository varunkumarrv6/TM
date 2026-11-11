#!/usr/bin/env bash
# Verification on the Pi: the clean Hamming stream, the stream with 1 % single-bit errors and the raw stream
# must all rebuild the SAME model (hash) and reproduce the TMU class sums of every test sample.
# Compare the printed numbers with data/expected.txt.
set -u
cd "$(dirname "$0")"
DATA="${1:-data}"
make -s all || exit 1
fail=0
for s in medsec_tm_hamming.bin medsec_tm_hamming_err.bin medsec_tm_raw.bin; do
    echo "== $s"
    ./bin/tm10_final_neon "$DATA" 0.2 "$s" | grep -v '^RESULT|' || fail=1
done
echo; echo "== expected (data/expected.txt)"; cat "$DATA/expected.txt"
[ $fail -eq 0 ] && echo "VERIFY: PASS" || echo "VERIFY: FAIL"
exit $fail
