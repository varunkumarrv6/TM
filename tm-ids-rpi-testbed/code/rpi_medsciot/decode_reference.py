"""Reference decoder + TM inference for the central node (numpy only, independent of the generator code).

usage:  python decode_reference.py [--raw] tm0_hamming.bin tm1_hamming.bin tm2_hamming.bin
Reads each TM's datagram stream (as written by make_rpi_files.py, or as captured at the central node after the
laptop/MITM), Hamming-decodes it, rebuilds the clause banks + 4-bit weights, classifies X_50.bin and compares with
tmu_pred.bin (deployed model, no attack) and actual.bin (ground truth).
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NUM_CLASSES, INPUT_FEATURES, RULE_BYTES, RULES_PER_PACKET, HEADER = 5, 178, 45, 12, 4
L = 2 * INPUT_FEATURES
RULE_AREA = RULES_PER_PACKET * RULE_BYTES                  # 540
PAYLOAD = RULE_AREA + HEADER                               # 544
ENC = PAYLOAD * 14 // 8                                    # 952
T = 33


def hamming_decode(enc):
    """952 bytes -> (544 bytes, corrected codewords). Codeword P1 P2 D1 P4 D2 D3 D4, MSB-first; nibble bit 3 = D1."""
    bits = np.unpackbits(np.frombuffer(enc, np.uint8), bitorder="big")[: PAYLOAD * 2 * 7].reshape(-1, 7).copy()
    s = (bits[:, 0] ^ bits[:, 2] ^ bits[:, 4] ^ bits[:, 6]) | ((bits[:, 1] ^ bits[:, 2] ^ bits[:, 5] ^ bits[:, 6]) << 1) \
        | ((bits[:, 3] ^ bits[:, 4] ^ bits[:, 5] ^ bits[:, 6]) << 2)
    bad = np.nonzero(s)[0]
    bits[bad, s[bad].astype(int) - 1] ^= 1
    nib = (bits[:, 2] << 3) | (bits[:, 4] << 2) | (bits[:, 5] << 1) | bits[:, 6]
    return ((nib[0::2] << 4) | nib[1::2]).astype(np.uint8), len(bad)


def parse_header(h):
    w = int(h[0]) >> 4
    return (w - 16 if w > 7 else w), int(h[0]) & 0x0F, int(h[1]) | (int(h[2]) << 8), int(h[3])


def decode_stream(path, raw=False):
    size = HEADER + (PAYLOAD if raw else ENC)
    data = np.fromfile(path, np.uint8)
    rules, weights, classes, corrected, hdr_mismatch, seqs = [], [], [], 0, 0, set()
    for s in range(0, data.size, size):
        dg = data[s:s + size]
        if raw:
            pay = dg[HEADER:]
        else:
            pay, c = hamming_decode(dg[HEADER:].tobytes())
            corrected += c
        hdr = pay[RULE_AREA:]                               # Hamming-protected copy of the header
        hdr_mismatch += int(not np.array_equal(hdr, dg[:HEADER]))
        w, cls, seq, node = parse_header(hdr)
        if seq in seqs:
            continue                                        # duplicate datagram
        seqs.add(seq)
        r = np.unpackbits(pay[:RULE_AREA].reshape(RULES_PER_PACKET, RULE_BYTES), axis=1, bitorder="little")[:, :L]
        rules.append(r); weights += [w] * RULES_PER_PACKET; classes += [cls] * RULES_PER_PACKET
    return np.concatenate(rules), np.array(weights), np.array(classes), corrected, hdr_mismatch, node, len(seqs)


def class_sums(X, rules, weights, classes):
    lit = np.concatenate([X, 1 - X], axis=1).astype(bool)                   # (N, 356)
    inc = rules.astype(bool)                                                 # (R, 356)
    fires = ~((inc[None, :, :]) & (~lit[:, None, :])).any(axis=2) & inc.any(axis=1)[None, :]
    out = np.zeros((X.shape[0], NUM_CLASSES), np.int64)
    for c in range(NUM_CLASSES):
        out[:, c] = (fires[:, classes == c] * weights[classes == c]).sum(axis=1)
    return out


if __name__ == "__main__":
    raw = "--raw" in sys.argv
    files = [a for a in sys.argv[1:] if not a.startswith("--")] or \
            [f"{HERE}/tm{n}_{'raw' if raw else 'hamming'}.bin" for n in range(3)]
    Xb = np.fromfile(f"{HERE}/X_50.bin", np.uint8).reshape(50, -1)
    X = np.unpackbits(Xb, axis=1, bitorder="little")[:, :INPUT_FEATURES]
    actual = np.fromfile(f"{HERE}/actual.bin", np.uint8)
    ref = np.fromfile(f"{HERE}/tmu_pred.bin", np.uint8)
    names = [l.split(" ", 1)[1].strip() for l in open(f"{HERE}/class_names.txt")]
    total = np.zeros((50, NUM_CLASSES), np.int64)
    for f in files:
        rules, w, cls, corr, hm, node, npk = decode_stream(f, raw)
        total += np.clip(class_sums(X, rules, w, cls), -T, T)
        print(f"{os.path.basename(f):22s} node {node}: {npk} packets, {corr:,} codewords corrected, "
              f"{hm} clear headers differ from the protected copy")
    pred = total.argmax(axis=1)
    print(f"\n{'#':>3} {'actual':18s} {'reference':18s} {'decoded':18s} match")
    for i in range(50):
        print(f"{i:3d} {names[actual[i]]:18s} {names[ref[i]]:18s} {names[pred[i]]:18s} {'YES' if pred[i] == ref[i] else 'NO'}")
    print(f"\ndecoded == reference : {(pred == ref).sum()}/50    accuracy decoded {(pred == actual).mean()*100:.1f} %   "
          f"reference {(ref == actual).mean()*100:.1f} %")
