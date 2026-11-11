#!/bin/bash
# Whole testbed on ONE machine (all addresses 127.0.0.1, laptop forwarder on port 5002) to check the programs
# before using the real RPis. Runs: clean, K random bits anywhere in TM 1's datagrams (toggle / include / exclude,
# Hamming and raw), and one literal-only attack (target literals, as in the RPi runs 6-12).
#   usage: bash test_localhost.sh        -> results_localhost/ + check against check_run.py
set -e
cd "$(dirname "$0")"
B=build_localhost
mkdir -p $B
L=(-DCENTRAL_IP='"127.0.0.1"' -DLAPTOP_IP='"127.0.0.1"' -DLAPTOP_PORT=5002 -DRPI0_IP='"127.0.0.1"'
   -DRPI1_IP='"127.0.0.1"' -DRPI2_IP='"127.0.0.1"' -DGAP_US=300)
if [ "$(uname -m)" = aarch64 ]; then A=-mcpu=native; else A="-march=native -msse4.1"; fi
gcc -O3 $A "${L[@]}" -o $B/receiver receiver.c
gcc -O2 "${L[@]}" -o $B/forwarder forwarder.c
for n in 0 1 2; do gcc -O2 -DNODE=$n "${L[@]}" -o $B/sender$n sender.c; done

rm -rf results_localhost
$B/receiver --out results_localhost --show 3 > $B/receiver.log 2>&1 &
RX=$!
sleep 0.5
run() {   # attack K raw|hamming [target]
    $B/forwarder -a "$1" -k "$2" -t "${4:-all}" > $B/forwarder.log 2>&1 &
    FW=$!
    sleep 0.3
    L3=""; [ "$3" = raw ] && L3=raw
    $B/sender0 $L3 > /dev/null & S0=$!
    $B/sender1 $L3 > /dev/null & S1=$!
    $B/sender2 $L3 > /dev/null & S2=$!
    wait $S0 $S1 $S2
    sleep 1
    kill $FW; wait $FW 2>/dev/null || true
    sleep 3.5                     # > forwarder's 2 s pause, so the next run gets a fresh label
}
run none 0 hamming
run toggle 4 hamming
run toggle 16 hamming
run include 64 hamming
run toggle 4 raw
run exclude 16 raw
run include 8 hamming literals
kill $RX; wait $RX 2>/dev/null || true
cat $B/receiver.log
echo
${PYTHON:-python3} check_run.py --runs results_localhost/runs.csv
