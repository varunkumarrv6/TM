"""Maps a random bit-error forwarder run to the FULL MedSec-25 test set (105,066 flows). Run on the machine that
has the model (/home/<user>/medsciot). Uses the REAL Hamming encode -> corrupt -> decode path, so parity-bit and
data-bit flips are handled exactly as the RPi central node does.

Give it the SAME --attack and --seed (and --bmin/--bmax) that forwarder_errors.py printed:
    python3 map_to_fulltest.py --attack toggle --seed 7
"""
import argparse
import os
import random
import sys
import numpy as np

sys.path.insert(0, "/home/<user>/medsciot")
import ecc_pipeline as E          # noqa: E402

RULE_CODEWORDS = 1080
PARITY_POS = {0, 1, 3}


def gen_positions(seed, cwmin, cwmax, bits_per_cw):
    r = random.Random(seed)
    n_cw = r.randint(cwmin, cwmax)
    codewords = sorted(r.sample(range(RULE_CODEWORDS), n_cw))
    positions = []
    for c in codewords:
        for b in sorted(r.sample(range(7), bits_per_cw)):
            positions.append(c * 7 + b)
    return n_cw, sorted(positions)


def corrupt_decode(n, positions, attack, M):
    """Encode TM n (SEQ), flip the given encoded-bit positions in every datagram, Hamming-decode, return cube+w."""
    path = E.encode_node_file(n, M["cubes"][n], M["weights"][n], "SEQ")
    data = np.fromfile(path, np.uint8).reshape(-1, E.DG_ENC)
    for p in positions:
        byte, mask = 4 + p // 8, np.uint8(0x80 >> (p % 8))
        if attack == "include":
            data[:, byte] |= mask
        elif attack == "exclude":
            data[:, byte] &= np.uint8(0xFF ^ int(mask))
        else:
            data[:, byte] ^= mask
    tmp = path.replace(".bin", "_corr.bin")
    data.reshape(-1).tofile(tmp)
    cube, w, _ = E.decode_node_file(tmp, n, "SEQ")
    os.remove(tmp)
    return cube, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", required=True, choices=["include", "exclude", "toggle"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--cwmin", type=int, default=1)
    ap.add_argument("--cwmax", type=int, default=10)
    ap.add_argument("--bits-per-cw", type=int, default=2)
    a = ap.parse_args()

    n_cw, positions = gen_positions(a.seed, a.cwmin, a.cwmax, a.bits_per_cw)
    parity = sum(1 for p in positions if p % 7 in PARITY_POS)
    coll = n_cw if a.bits_per_cw >= 2 else 0

    M = E.load_nodes(); T = M["T"]; y = E.yte
    clean = M["clean_sums"]
    CLEAN = E.fuse(clean, T)
    BENCH = (CLEAN == y).mean() * 100
    idx = np.load("/home/<user>/rpi_medsciot/test_indices_in_medsec_testset.npy")

    print("=" * 76)
    print(f"attack {a.attack.upper()}  seed {a.seed}  codewords={n_cw}  bits_per_cw={a.bits_per_cw}  "
          f"total_bits={len(positions)} (parity {parity}, data {len(positions)-parity})")
    print(f"uncorrectable codewords (>=2 flips in one codeword): {coll}")
    print("=" * 76)
    print(f"benchmark (no attack): full {BENCH:.4f} %   |   50-sample {(CLEAN[idx] == y[idx]).mean()*100:.2f} %\n")

    corrupted = {n: corrupt_decode(n, positions, a.attack, M) for n in range(E.N_NODES)}

    def report(label, nodes):
        s = [E.class_sums_all(*corrupted[n]) if n in nodes else clean[n] for n in range(E.N_NODES)]
        pred = E.fuse(s, T)
        acc, acc50 = (pred == y).mean() * 100, (pred[idx] == y[idx]).mean() * 100
        agr, agr50 = (pred == CLEAN).mean() * 100, (pred[idx] == CLEAN[idx]).mean() * 100
        print(f"{label}")
        print(f"   FULL accuracy {acc:7.4f} %   drop {acc - BENCH:+8.4f} pp   decoded==reference {agr:6.2f} %")
        print(f"   50   accuracy {acc50:7.2f} %   decoded==reference {agr50:6.2f} % "
              f"({int(round(agr50/100*50))}/50)\n")

    report("only TM2 attacked  (RPi hardware topology)", {2})
    report("all 3 TMs attacked (if all routed through the laptop)", {0, 1, 2})

    if coll == 0:
        print("note: 0 colliding codewords -> every flip (parity or data) is a single-bit error that Hamming")
        print("      repairs, so the model is restored exactly and accuracy is unchanged (PASS).")
    else:
        print(f"note: {coll} codeword(s) got >=2 flips -> uncorrectable; those are what move the accuracy.")
        print("      parity-only single flips never hurt (Hamming fixes them); damage needs >=2 flips in one codeword.")


if __name__ == "__main__":
    main()
