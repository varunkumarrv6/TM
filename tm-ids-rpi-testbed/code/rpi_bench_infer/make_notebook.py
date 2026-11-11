"""Writes benchmark_rpi.ipynb (and benchmark_rpi_cells.py, the same code as a plain script for testing).
usage: python make_notebook.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
cells = []
md = lambda s: cells.append(("markdown", s.strip("\n")))
code = lambda s: cells.append(("code", s.strip("\n")))

md(r"""
# TM inference benchmark on the Raspberry Pi — Python → NumPy → C / NEON → cores → pipelining

MedSec-25 single Tsetlin Machine (5 classes × 900 clauses, 356 literals, **4-bit normalised weights**).
The model arrives as a stream of Hamming(7,4)-encoded UDP-style datagrams (`data/medsec_tm_hamming.bin`).
Every method below **decodes the stream, rebuilds the clause bank and classifies the same 256 test flows**,
and every method is checked against the reference class sums (`data/scores_ref.npy`).

| step | methods |
|---|---|
| 1. pure Python (standard library) | Hamming decode + rebuild; inference: unpacked lists, packed bytes, packed + early exit, big-int bitmasks |
| 2. NumPy | vectorised Hamming decode; inference: unpacked, packed uint8, packed uint64, batched uint64 |
| 3. C (gcc -O3) | 10 kernels: scalar → 64-bit → NEON → **clause parallelism** (16 clauses / vector) → **core parallelism** (OpenMP) → final NEON |
| 4. cores | OpenMP kernel at 1–4 threads vs single-core kernels |
| 5. pipelining | decode / correct / infer per packet: sequential, streamed, software-pipelined (1 core), 3-thread pipeline, hybrids (OpenMP decode+ECC then 1-core inference; 3 decode+ECC workers + 1 inference core) |
| 6. end to end | first datagram → 256 predictions, split into decode / ECC / inference, for every approach |

**Run:** *Kernel → Restart & Run All* (or `./run_notebook.sh`). Results go to `results_rpi/`
(figures as PDF + PNG, `all_results.csv`, `stage_breakdown.csv`, `summary.txt`). Needs `python3-numpy`, `python3-matplotlib`, `gcc`, `make`.
""")

code(r"""
# ---------------- settings ----------------
N_PY       = 16      # test flows for the pure-Python kernels (slow); NumPy and C use all 256
PY_REPS    = 3       # timed repeats for each Python / NumPy measurement (median)
C_BUDGET   = 2.0     # seconds of timed runs per C kernel
THREAD_REPS = 5      # runs per thread count
PIPE_BUDGET = 2.0    # seconds per pipeline mode
RUN_C      = True    # build and run the C benchmarks
# ------------------------------------------

import os, sys, re, time, platform, subprocess, csv, statistics
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

HERE = os.path.abspath(".")
if not os.path.exists(os.path.join(HERE, "data")) and "__file__" in globals():    # script started elsewhere
    HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "results_rpi")
os.makedirs(OUT, exist_ok=True)
assert os.path.exists(os.path.join(DATA, "medsec_tm_hamming.bin")), "run this notebook inside rpi_bench_infer/"

DIM = {k: int(v, 0) for k, v in re.findall(r"#define\s+(\w+)\s+(0x[0-9a-fA-F]+|\d+)", open("model_dims.h").read())}
C_, L, O, RB = DIM["NUM_CLASSES"], DIM["LITERALS"], DIM["FEATURES"], DIM["RULE_BYTES"]
HDR, PAY, ENC, NPK, NS = DIM["HDR_BYTES"], DIM["PAYLOAD_BYTES"], DIM["ENC_BYTES"], DIM["N_PACKETS"], DIM["NS"]
MODEL_HASH = DIM["MODEL_HASH"]


def cpu_model():
    try:
        txt = open("/proc/cpuinfo").read()
        m = re.search(r"^(?:Model|model name)\s*:\s*(.+)$", txt, re.M)
        return m.group(1).strip() if m else platform.processor()
    except OSError:
        return platform.processor()


def governor():
    try:
        return open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read().strip()
    except OSError:
        return "n/a"


HOST = f"{cpu_model()} | {platform.machine()} | {os.cpu_count()} cores | governor {governor()} | Python {platform.python_version()} | NumPy {np.__version__}"
print(HOST)
if governor() not in ("performance", "n/a"):
    print("TIP: for stable numbers run  echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")
RESULTS = []        # rows: family, method, cores, receive_ms, infer_us_per_sample, ok


def timed(fn, reps):
    out, ts = None, []
    for _ in range(reps):
        t0 = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t0)
    return out, statistics.median(ts)
""")

md("## Data: test flows, references and the datagram stream")

code(r"""
X = np.load(f"{DATA}/X_test.npy")
y_true = np.load(f"{DATA}/y_true.npy").astype(int)
y_ref = np.load(f"{DATA}/y_ref.npy").astype(int)
scores_ref = np.load(f"{DATA}/scores_ref.npy").astype(np.int64)
STREAM = open(f"{DATA}/medsec_tm_hamming.bin", "rb").read()
DG = HDR + ENC
assert len(STREAM) == NPK * DG
lit = np.concatenate([X, 1 - X], 1).astype(np.uint8)                 # literal vectors [x, 1-x]
lit_p = np.packbits(np.pad(lit, ((0, 0), (0, RB * 8 - L))), axis=1, bitorder="little")
print(f"{NS} test flows x {O} features, {NPK} datagrams x {DG} bytes, reference accuracy {np.mean(y_ref == y_true) * 100:.2f} %")
""")

md(r"""
## 1. Pure Python (standard library only)
### Receive: Hamming(7,4) decode with the formula-based syndrome + header checks + rebuild
Codeword bits `P1 P2 D1 P4 D2 D3 D4`; `s1 = P1⊕D1⊕D2⊕D4`, `s2 = P2⊕D1⊕D3⊕D4`, `s4 = P4⊕D2⊕D3⊕D4`; a non-zero
syndrome flips bit `7 − s`. The header carries the class, the number of real rules, the sequence number,
the **normalisation flag + 4-bit weight** and the scale `s` (Q8.8).
""")

code(r"""
def syndrome(cw):
    p1, p2, d1, p4 = (cw >> 6) & 1, (cw >> 5) & 1, (cw >> 4) & 1, (cw >> 3) & 1
    d2, d3, d4 = (cw >> 2) & 1, (cw >> 1) & 1, cw & 1
    return (p1 ^ d1 ^ d2 ^ d4) | ((p2 ^ d1 ^ d3 ^ d4) << 1) | ((p4 ^ d2 ^ d3 ^ d4) << 2)


def nibble(cw):                                           # D1 D2 D3 D4 -> nibble bits 3..0
    return (((cw >> 4) & 1) << 3) | (((cw >> 2) & 1) << 2) | (((cw >> 1) & 1) << 1) | (cw & 1)


def py_detect(buf):                       # DECODE: codewords + formula-based syndromes of every datagram
    out = []
    for s in range(NPK):
        enc = buf[s * DG + HDR:(s + 1) * DG]
        cws, syns = [], []
        for g in range(PAY // 4):
            v = int.from_bytes(enc[7 * g:7 * g + 7], "big")
            for i in range(8):
                cw = (v >> (49 - 7 * i)) & 0x7F
                cws.append(cw)
                syns.append(syndrome(cw))
        out.append((cws, syns))
    return out


def py_correct(buf, det):                 # ECC: flip bit 7 - s, rebuild payload, checks, place the rules
    rules, weights = [[] for _ in range(C_)], [[] for _ in range(C_)]
    fixed, scale, normalised = 0, None, True
    for s in range(NPK):
        h = buf[s * DG:s * DG + HDR]
        cws, syns = det[s]
        pay = bytearray(PAY)
        for j in range(PAY):
            hi, lo, sh, sl = cws[2 * j], cws[2 * j + 1], syns[2 * j], syns[2 * j + 1]
            if sh:
                hi ^= 1 << (7 - sh); fixed += 1
            if sl:
                lo ^= 1 << (7 - sl); fixed += 1
            pay[j] = (nibble(hi) << 4) | nibble(lo)
        assert pay[PAY - HDR:] == h and (h[2] | h[3] << 8) == s and (h[7] & 1) == 1, f"packet {s} failed"
        normalised &= bool(h[4] & 0x80)
        scale = (h[5] | h[6] << 8) / 256
        w4 = h[4] & 0x0F
        w4 = w4 - 16 if w4 > 7 else w4
        for r in range(h[1]):
            rules[h[0]].append(bytes(pay[r * RB:(r + 1) * RB]))
            weights[h[0]].append(w4)
    return rules, weights, fixed, normalised, scale


def model_hash(rules, weights):
    hsh = 0xcbf29ce484222325
    for c in range(C_):
        for rule, w in zip(rules[c], weights[c]):
            for b in rule + (w & 0xFFFF).to_bytes(2, "little"):
                hsh = ((hsh ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return hsh


DET_PY, t_py_det = timed(lambda: py_detect(STREAM), 1)
(rules_b, weights_l, fixed, normalised, scale), t_py_cor = timed(lambda: py_correct(STREAM, DET_PY), 1)
t_py_rx = t_py_det + t_py_cor
hash_ok = model_hash(rules_b, weights_l) == MODEL_HASH
print(f"pure-Python receive: {t_py_rx * 1e3:.1f} ms = decode {t_py_det * 1e3:.1f} + ECC {t_py_cor * 1e3:.1f} ms, "
      f"corrected {fixed}, weights normalised={normalised}, s={scale:.4f}, "
      f"hash {'OK' if hash_ok else 'BAD'}, clauses/class {[len(r) for r in rules_b]}")
assert hash_ok and normalised
""")

md("### Inference: four pure-Python kernels (on the first `N_PY` test flows)")

code(r"""
lits_u = [list(map(int, lit[s])) for s in range(NS)]
lits_b = [bytes(lit_p[s]) for s in range(NS)]
lits_i = [int.from_bytes(b, "little") for b in lits_b]
rules_u = [[[(rule[i // 8] >> (i % 8)) & 1 for i in range(L)] for rule in rules_b[c]] for c in range(C_)]
rules_i = [[int.from_bytes(rule, "little") for rule in rules_b[c]] for c in range(C_)]


def py_unpacked(s):                      # one 0/1 per literal, every literal checked
    x = lits_u[s]
    out = []
    for c in range(C_):
        tot = 0
        for rule, w in zip(rules_u[c], weights_l[c]):
            viol = nonempty = 0
            for r, v in zip(rule, x):
                viol |= r & (v ^ 1)
                nonempty |= r
            if nonempty and not viol:
                tot += w
        out.append(tot)
    return out


def py_packed(s):                        # 45 packed bytes, all bytes checked
    x = lits_b[s]
    out = []
    for c in range(C_):
        tot = 0
        for rule, w in zip(rules_b[c], weights_l[c]):
            miss = nonempty = 0
            for r, v in zip(rule, x):
                miss |= r & ~v & 0xFF
                nonempty |= r
            if nonempty and not miss:
                tot += w
        out.append(tot)
    return out


def py_early_exit(s):                    # packed bytes, stop at the first failing byte
    x = lits_b[s]
    out = []
    for c in range(C_):
        tot = 0
        for rule, w in zip(rules_b[c], weights_l[c]):
            for r, v in zip(rule, x):
                if r & ~v & 0xFF:
                    break
            else:
                if any(rule):
                    tot += w
        out.append(tot)
    return out


def py_bigint(s):                        # one Python integer per rule: fires iff rule & ~x == 0
    x = lits_i[s]
    return [sum(w for r, w in zip(rules_i[c], weights_l[c]) if r and not (r & ~x)) for c in range(C_)]


PY_KERNELS = [("Python unpacked", py_unpacked), ("Python packed bytes", py_packed),
              ("Python packed early-exit", py_early_exit), ("Python big-int", py_bigint)]
for name, fn in PY_KERNELS:
    sc, t = timed(lambda: [fn(s) for s in range(N_PY)], PY_REPS)
    ok = np.array_equal(np.array(sc), scores_ref[:N_PY])
    RESULTS.append(dict(family="Python", method=name, cores=1, receive_ms=t_py_rx * 1e3,
                        infer_us_per_sample=t / N_PY * 1e6, ok=int(ok and hash_ok)))
    print(f"{name:28s} {t / N_PY * 1e3:10.3f} ms/sample   scores==ref {ok}")
""")

md(r"""
## 2. NumPy
### Receive: vectorised Hamming(7,4) decode of all datagrams at once
""")

code(r"""
def np_detect(buf):                       # DECODE: all codewords as bit arrays + syndromes, all datagrams at once
    a = np.frombuffer(buf, np.uint8).reshape(NPK, DG)
    cw = np.unpackbits(a[:, HDR:], axis=1, bitorder="big")[:, : PAY * 2 * 7].reshape(NPK, PAY * 2, 7)
    s = (cw[..., 0] ^ cw[..., 2] ^ cw[..., 4] ^ cw[..., 6]) | ((cw[..., 1] ^ cw[..., 2] ^ cw[..., 5] ^ cw[..., 6]) << 1) \
        | ((cw[..., 3] ^ cw[..., 4] ^ cw[..., 5] ^ cw[..., 6]) << 2)
    return a[:, :HDR], cw, s


def np_correct(hdr, cw, s):               # ECC: flip bit 7 - s, rebuild payloads, checks, place the rules
    cw = cw.copy()
    pk, ci = np.nonzero(s)
    cw[pk, ci, s[pk, ci].astype(np.int64) - 1] ^= 1
    nib = (cw[..., 2] << 3) | (cw[..., 4] << 2) | (cw[..., 5] << 1) | cw[..., 6]
    pay = ((nib[:, 0::2] << 4) | nib[:, 1::2]).astype(np.uint8)
    assert np.array_equal(pay[:, PAY - HDR:], hdr) and np.array_equal(hdr[:, 2].astype(int) | hdr[:, 3].astype(int) << 8, np.arange(NPK))
    w4 = (hdr[:, 4] & 0x0F).astype(np.int64); w4[w4 > 7] -= 16
    rules = [[] for _ in range(C_)]; weights = [[] for _ in range(C_)]
    for p in range(NPK):
        c, n = int(hdr[p, 0]), int(hdr[p, 1])
        rules[c].append(pay[p, : n * RB].reshape(n, RB)); weights[c] += [w4[p]] * n
    CLP = max(sum(len(r) for r in rules[c]) for c in range(C_))
    R = np.zeros((C_, CLP, RB), np.uint8); W = np.zeros((C_, CLP), np.int64)
    for c in range(C_):
        rr = np.concatenate(rules[c]); R[c, :len(rr)] = rr; W[c, :len(rr)] = weights[c]
    return R, W, len(pk)


DET_NP, t_np_det = timed(lambda: np_detect(STREAM), PY_REPS)
(R8, W, fixed_np), t_np_cor = timed(lambda: np_correct(*DET_NP), PY_REPS)
t_np_rx = t_np_det + t_np_cor
np_ok_model = all(R8[c, :len(rules_b[c])].tobytes() == b"".join(rules_b[c]) for c in range(C_))
print(f"NumPy receive: {t_np_rx * 1e3:.2f} ms = decode {t_np_det * 1e3:.2f} + ECC {t_np_cor * 1e3:.2f} ms, "
      f"corrected {fixed_np}, model == pure-Python model: {np_ok_model}")
""")

md("### Inference: four NumPy kernels (all 256 test flows)")

code(r"""
U = np.unpackbits(R8, axis=2, bitorder="little")[..., :L]          # (C, CLP, 356) unpacked
NE = R8.any(2)                                                      # non-empty clauses
R64 = np.pad(R8, ((0, 0), (0, 0), (0, 48 - RB))).view(np.uint64)    # (C, CLP, 6)
x64 = np.pad(lit_p, ((0, 0), (0, 48 - RB))).view(np.uint64)         # (NS, 6)


def np_unpacked():                       # one byte per literal, per test flow over all clauses
    out = []
    for s in range(NS):
        fires = ~(U & (1 - lit[s])[None, None]).any(2) & NE
        out.append((fires * W).sum(1))
    return np.stack(out)


def np_packed_u8():                      # 45 packed bytes per clause
    out = []
    for s in range(NS):
        fires = ~(R8 & ~lit_p[s][None, None]).any(2) & NE
        out.append((fires * W).sum(1))
    return np.stack(out)


def np_packed_u64():                     # 6 x 64-bit words per clause
    out = []
    for s in range(NS):
        fires = ~(R64 & ~x64[s][None, None]).any(2) & NE
        out.append((fires * W).sum(1))
    return np.stack(out)


def np_batched_u64(chunk=64):
    out = []
    for a in range(0, NS, chunk):
        miss = (R64[None] & ~x64[a:a + chunk, None, None]).any(3)          # (chunk, C, CLP)
        out.append(((~miss & NE[None]) * W[None]).sum(2))
    return np.concatenate(out)


NP_KERNELS = [("NumPy unpacked", np_unpacked), ("NumPy packed uint8", np_packed_u8),
              ("NumPy packed uint64", np_packed_u64), ("NumPy batched uint64", np_batched_u64)]
for name, fn in NP_KERNELS:
    sc, t = timed(fn, PY_REPS)
    ok = np.array_equal(sc, scores_ref)
    RESULTS.append(dict(family="NumPy", method=name, cores=1, receive_ms=t_np_rx * 1e3,
                        infer_us_per_sample=t / NS * 1e6, ok=int(ok and np_ok_model)))
    print(f"{name:28s} {t / NS * 1e3:10.4f} ms/sample   scores==ref {ok}")
""")

md(r"""
## 3–5. C: NEON, clause parallelism, core parallelism, pipelining
Builds the kernels with `make` (on the Pi: real NEON, `-mcpu=native`) and runs `run_all.sh`, `run_threads.sh`
and `run_pipeline.sh`. Each program decodes the Hamming stream itself and checks hash, class sums and predictions.
""")

code(r"""
def sh(cmd):
    print("$", " ".join(cmd))
    p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    print(p.stdout[-4000:])
    if p.returncode != 0:
        print(p.stderr[-2000:])
    return p.returncode


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


C_ROWS, THR_ROWS, PIPE_ROWS = [], [], []
if RUN_C:
    sh(["make", "-s", "all"])
    sh(["bash", "run_all.sh", "data", str(C_BUDGET)])               # via bash: works even without +x
    sh(["bash", "run_threads.sh", "data", "1.0", str(THREAD_REPS)])
    sh(["bash", "run_pipeline.sh", "data", str(PIPE_BUDGET)])
    C_ROWS, THR_ROWS, PIPE_ROWS = read_csv("results.csv"), read_csv("threads_results.csv"), read_csv("pipeline_results.csv")
    for r in C_ROWS:
        RESULTS.append(dict(family="C", method="C " + r["kernel"], cores=int(r["threads"]),
                            receive_ms=float(r["receive_ms"]), infer_us_per_sample=float(r["infer_ms_per_sample"]) * 1e3,
                            ok=int(r["all_ok"])))
print(f"{len(C_ROWS)} C kernels, {len(THR_ROWS)} thread runs, {len(PIPE_ROWS)} pipeline rows")
""")

md(r"""
## 6. End to end: first datagram → predictions of all 256 flows, split into decode / ECC / inference
**decode** = codewords + formula syndromes (detection), **ECC** = bit correction + payload rebuild + header checks
+ placing the rules, **inference** = kernel preparation + class sums of the 256 flows + argmax.
For the sequential methods the end-to-end time is one wall-clock run of all three stages; in the pipelines the
stages overlap, so the end-to-end time is *shorter* than the sum of the stage (busy) times.
""")

code(r"""
def py_end_to_end():                          # pure Python, big-int kernel (fastest pure-Python inference)
    t0 = time.perf_counter()
    det = py_detect(STREAM)
    t1 = time.perf_counter()
    rb, wl, *_ = py_correct(STREAM, det)
    t2 = time.perf_counter()
    ri = [[int.from_bytes(r, "little") for r in rb[c]] for c in range(C_)]
    pred = []
    for s in range(NS):
        x = lits_i[s]
        sc = [sum(w for r, w in zip(ri[c], wl[c]) if r and not (r & ~x)) for c in range(C_)]
        pred.append(sc.index(max(sc)))
    t3 = time.perf_counter()
    return pred, (t1 - t0, t2 - t1, t3 - t2, t3 - t0)


def np_end_to_end():                          # NumPy, packed uint64 kernel
    t0 = time.perf_counter()
    det = np_detect(STREAM)
    t1 = time.perf_counter()
    R, Wn, _ = np_correct(*det)
    t2 = time.perf_counter()
    R64e = np.pad(R, ((0, 0), (0, 0), (0, 48 - RB))).view(np.uint64)
    NEe = R.any(2)
    sc = np.stack([((~(R64e & ~x64[s][None, None]).any(2) & NEe) * Wn).sum(1) for s in range(NS)])
    pred = sc.argmax(1)
    t3 = time.perf_counter()
    return pred, (t1 - t0, t2 - t1, t3 - t2, t3 - t0)


def median_run(fn, reps):
    runs = [fn() for _ in range(reps)]
    t = np.median(np.array([r[1] for r in runs]), axis=0)
    ok = all(np.array_equal(np.asarray(r[0]), y_ref) for r in runs)
    return t * 1e3, ok


STAGE_ROWS = []
t, ok = median_run(py_end_to_end, 1)
STAGE_ROWS.append(dict(method="pure Python (big-int)", cores=1, decode_ms=t[0], ecc_ms=t[1], infer_ms=t[2], e2e_ms=t[3], ok=int(ok)))
t, ok = median_run(np_end_to_end, PY_REPS)
STAGE_ROWS.append(dict(method="NumPy (packed uint64)", cores=1, decode_ms=t[0], ecc_ms=t[1], infer_ms=t[2], e2e_ms=t[3], ok=int(ok)))
if C_ROWS:
    for k, lab in (("10", "C final NEON"), ("09", "C clause-par. NEON + OpenMP")):
        r = next(r for r in C_ROWS if r["kernel"].startswith(k))
        STAGE_ROWS.append(dict(method=f"{lab} ({r['threads']} core{'s' if r['threads'] != '1' else ''})",
                               cores=int(r["threads"]), decode_ms=float(r["detect_ms"]), ecc_ms=float(r["correct_ms"]),
                               infer_ms=float(r["prep_ms"]) + float(r["batch_ms"]), e2e_ms=float(r["e2e_ms"]),
                               ok=int(r["all_ok"])))
if PIPE_ROWS:
    for mo, lab in (("swp", "C pipeline swp"), ("pipe", "C pipeline pipe"),
                    ("hyb2ph", "C hybrid: OpenMP decode+ECC, then 1-core infer"),
                    ("hybpipe", "C hybrid pipeline: 3 decode+ECC workers + 1 infer core")):
        r = next(r for r in PIPE_ROWS if r["mode"] == mo and r["stream"] == "medsec_tm_hamming.bin")
        STAGE_ROWS.append(dict(method=f"{lab} ({r['cores']} core{'s' if r['cores'] != '1' else ''})",
                               cores=int(r["cores"]), decode_ms=float(r["decode_busy_ms"]), ecc_ms=float(r["ecc_busy_ms"]),
                               infer_ms=float(r["infer_busy_ms"]), e2e_ms=float(r["e2e_ms"]), ok=int(r["all_ok"])))
with open(f"{OUT}/stage_breakdown.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(STAGE_ROWS[0].keys())); w.writeheader(); w.writerows(STAGE_ROWS)
print(f"{'method':58s} {'decode ms':>10s} {'ECC ms':>10s} {'infer ms':>10s} {'sum ms':>10s} {'end-to-end ms':>14s}  check")
for r in STAGE_ROWS:
    print(f"{r['method']:58s} {r['decode_ms']:10.3f} {r['ecc_ms']:10.3f} {r['infer_ms']:10.3f} "
          f"{r['decode_ms'] + r['ecc_ms'] + r['infer_ms']:10.3f} {r['e2e_ms']:14.3f}  {'OK' if r['ok'] else 'FAILED'}")
""")

md("## Results table and verification")

code(r"""
with open(f"{OUT}/all_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(RESULTS[0].keys())); w.writeheader(); w.writerows(RESULTS)
base = RESULTS[0]["infer_us_per_sample"]
lines = [f"host: {HOST}", "", f"{'method':42s} {'cores':>5s} {'receive ms':>11s} {'infer us/sample':>16s} {'speed-up':>9s}  check"]
for r in RESULTS:
    lines.append(f"{r['method']:42s} {r['cores']:5d} {r['receive_ms']:11.3f} {r['infer_us_per_sample']:16.3f} "
                 f"{base / r['infer_us_per_sample']:8.1f}x  {'OK' if r['ok'] else 'FAILED'}")
best = min(RESULTS, key=lambda r: r["infer_us_per_sample"])
lines += ["", f"fastest inference: {best['method']} ({best['cores']} core(s)), {best['infer_us_per_sample']:.3f} us/sample",
          f"all methods verified: {all(r['ok'] for r in RESULTS)}"]
if PIPE_ROWS:
    lines += ["", "pipeline (end-to-end: first packet -> class sums of 256 flows):"]
    for r in PIPE_ROWS:
        lines.append(f"  {r['stream']:28s} {r['mode']:7s} {r['cores']} core(s) {float(r['e2e_ms']):9.3f} ms  "
                     f"({float(r['speedup_vs_seq']):.2f}x vs seq)  {'OK' if r['all_ok'] == '1' else 'FAILED'}")
if STAGE_ROWS:
    lines += ["", "end to end (first datagram -> predictions of all 256 flows), stage times in ms:",
              f"  {'method':58s} {'decode':>9s} {'ECC':>9s} {'infer':>9s} {'end-to-end':>11s}"]
    for r in STAGE_ROWS:
        lines.append(f"  {r['method']:58s} {r['decode_ms']:9.3f} {r['ecc_ms']:9.3f} {r['infer_ms']:9.3f} "
                     f"{r['e2e_ms']:11.3f}  {'OK' if r['ok'] else 'FAILED'}")
    lines.append(f"  all end-to-end runs verified: {all(r['ok'] for r in STAGE_ROWS)}")
open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
""")

md("## Figures (saved to `results_rpi/` as PDF and PNG)")

code(r"""
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 7.5, "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
                     "axes.titlelocation": "left", "xtick.direction": "in", "ytick.direction": "in",
                     "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
FAM = {"Python": "#D55E00", "NumPy": "#E69F00", "C": "#0072B2"}
GREEN, ORANGE, GREY = "#009E73", "#CC79A7", "#7F7F7F"


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png", dpi=200); plt.show()


# 1: inference latency of every method
fig, ax = plt.subplots(figsize=(7.16, 3.0))
vals = [r["infer_us_per_sample"] for r in RESULTS]
ib = int(np.argmin(vals))
cols = [GREEN if i == ib else (ORANGE if r["cores"] > 1 else FAM[r["family"]]) for i, r in enumerate(RESULTS)]
x = np.arange(len(RESULTS))
ax.bar(x, vals, 0.72, color=cols)
for xi, v, r in zip(x, vals, RESULTS):
    ax.text(xi, v * 1.12, f"{v:.3g}", ha="center", va="bottom", fontsize=6, rotation=90)
ax.set_yscale("log"); ax.set_ylim(min(vals) / 2, max(vals) * 30)
ax.set_xticks(x, [r["method"].replace("C ", "").replace("Python ", "Py ").replace("NumPy ", "NP ") +
                  (f" ({r['cores']}T)" if r["cores"] > 1 else "") for r in RESULTS], rotation=60, ha="right", fontsize=6.5)
ax.set_ylabel(r"inference latency ($\mu$s / sample, log)")
ax.set_title(f"(a) TM inference latency on {cpu_model()}")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=FAM["Python"], label="pure Python"), Patch(color=FAM["NumPy"], label="NumPy"),
                   Patch(color=FAM["C"], label="C (1 core)"), Patch(color=ORANGE, label="C + OpenMP (multi-core)"),
                   Patch(color=GREEN, label=f"fastest: {RESULTS[ib]['method']}")], loc="upper right", ncol=2)
ax.grid(axis="x", visible=False)
save(fig, "fig1_inference_latency")

# 2: receive (Hamming decode + rebuild) and end-to-end (receive + 256 inferences)
rx = [("pure Python", t_py_rx * 1e3, 1, FAM["Python"]), ("NumPy", t_np_rx * 1e3, 1, FAM["NumPy"])]
if C_ROWS:
    c10 = next(r for r in C_ROWS if r["kernel"].startswith("10"))
    c09 = next(r for r in C_ROWS if r["kernel"].startswith("09"))
    rx += [("C (1 core)", float(c10["receive_ms"]), 1, FAM["C"]),
           (f"C OpenMP ({c09['threads']} cores)", float(c09["receive_ms"]), int(c09["threads"]), ORANGE)]
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.6))
ax = axes[0]
ax.bar(range(len(rx)), [r[1] for r in rx], 0.6, color=[r[3] for r in rx])
for i, r in enumerate(rx):
    ax.text(i, r[1] * 1.12, f"{r[1]:.3g} ms", ha="center", va="bottom", fontsize=7)
ax.set_yscale("log"); ax.set_ylim(top=max(r[1] for r in rx) * 6); ax.set_xticks(range(len(rx)), [r[0] for r in rx], rotation=15)
ax.set_ylabel("receive time (ms, log)"); ax.set_title(f"(b) Hamming decode + rebuild of {NPK} datagrams")
ax.grid(axis="x", visible=False)
ax = axes[1]
best_py = min((r for r in RESULTS if r["family"] == "Python"), key=lambda r: r["infer_us_per_sample"])
best_np = min((r for r in RESULTS if r["family"] == "NumPy"), key=lambda r: r["infer_us_per_sample"])
e2e = [("pure Python", t_py_rx * 1e3 + best_py["infer_us_per_sample"] * NS / 1e3, FAM["Python"]),
       ("NumPy", t_np_rx * 1e3 + best_np["infer_us_per_sample"] * NS / 1e3, FAM["NumPy"])]
if C_ROWS:
    e2e.append(("C NEON (1 core)", float(c10["receive_ms"]) + float(c10["prep_ms"]) + float(c10["batch_ms"]), FAM["C"]))
if PIPE_ROWS:
    for mode, col in (("swp", GREY), ("pipe", GREEN)):
        r = next(r for r in PIPE_ROWS if r["mode"] == mode and r["stream"] == "medsec_tm_hamming.bin")
        e2e.append((f"C pipeline {mode} ({r['cores']} core{'s' if r['cores'] != '1' else ''})", float(r["e2e_ms"]), col))
ax.bar(range(len(e2e)), [r[1] for r in e2e], 0.6, color=[r[2] for r in e2e])
for i, r in enumerate(e2e):
    ax.text(i, r[1] * 1.12, f"{r[1]:.3g} ms", ha="center", va="bottom", fontsize=7)
ax.set_yscale("log"); ax.set_ylim(top=max(r[1] for r in e2e) * 6); ax.set_xticks(range(len(e2e)), [r[0] for r in e2e], rotation=20, ha="right")
ax.set_ylabel("time (ms, log)"); ax.set_title(f"(c) model stream in -> {NS} predictions out")
ax.grid(axis="x", visible=False)
fig.tight_layout(w_pad=1.0)
save(fig, "fig2_receive_end_to_end")

# 3: threads and pipeline modes
if THR_ROWS and PIPE_ROWS:
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5))
    ax = axes[0]
    med = {}
    for r in THR_ROWS:
        med.setdefault((r["kernel"], int(r["threads"])), []).append(float(r["infer_ms_per_sample"]) * 1e3)
    t9 = sorted((t, statistics.median(v)) for (k, t), v in med.items() if k == "tm09_interclause_omp")
    ax.plot([a for a, _ in t9], [b for _, b in t9], "-o", color=ORANGE, label="09 clause-parallel NEON + OpenMP")
    for k, col, lab in (("tm08_interclause_neon", FAM["C"], "08 clause-parallel NEON, 1 core"),
                        ("tm10_final_neon", GREEN, "10 final NEON, 1 core")):
        v = statistics.median(med[(k, 1)])
        ax.axhline(v, color=col, ls="--", lw=1.1, label=f"{lab}: {v:.2f}")
    ax.set(xticks=[1, 2, 3, 4], xlabel="OpenMP threads (cores)", ylabel=r"$\mu$s / sample", title="(d) core parallelism")
    ax.legend(fontsize=6.5)
    ax = axes[1]
    streams = ["medsec_tm_hamming.bin", "medsec_tm_hamming_err.bin", "medsec_tm_raw.bin"]
    labels = ["Hamming", "+ 1 % bit errors", "raw (no ECC)"]
    modes = ["seq", "stream", "swp", "pipe", "hyb2ph", "hybpipe"]
    mcol = {"seq": GREY, "stream": FAM["NumPy"], "swp": FAM["C"], "pipe": GREEN, "hyb2ph": ORANGE, "hybpipe": "#56B4E9"}
    ncores = {r["mode"]: r["cores"] for r in PIPE_ROWS}
    wdt = 0.13
    for j, mo in enumerate(modes):
        v = [float(next(r for r in PIPE_ROWS if r["mode"] == mo and r["stream"] == s)["e2e_ms"]) for s in streams]
        ax.bar(np.arange(3) + (j - 2.5) * wdt, v, wdt * 0.95, color=mcol[mo],
               label=f"{mo} ({ncores[mo]} core{'s' if ncores[mo] != '1' else ''})")
    ax.set(xticks=range(3), ylabel="end-to-end (ms)", title="(e) pipelining: first packet -> 256 predictions")
    ax.set_xticklabels(labels)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.7)
    ax.legend(fontsize=6, ncol=3, loc="upper center"); ax.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig3_cores_pipeline")
# 4: end-to-end time split into decode / ECC / inference
if STAGE_ROWS:
    SC = {"decode": "#56B4E9", "ECC": "#E69F00", "inference": "#009E73"}
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0), gridspec_kw=dict(width_ratios=[1.2, 1.25]))
    ax = axes[0]
    crow = [r for r in STAGE_ROWS if r["method"].startswith("C ")]
    x = np.arange(len(crow))
    bottom = np.zeros(len(crow))
    for key, lab in (("decode_ms", "decode"), ("ecc_ms", "ECC"), ("infer_ms", "inference")):
        v = np.array([r[key] for r in crow])
        ax.bar(x, v, 0.6, bottom=bottom, color=SC[lab], label=lab)
        bottom += v
    e = np.array([r["e2e_ms"] for r in crow])
    ax.plot(x, e, "D", color="k", ms=6, label="end-to-end (measured)")
    for xi, v in zip(x, e):
        ax.text(xi + 0.32, v, f"{v:.2f}", va="center", fontsize=7)
    short = lambda m: (m.replace("C hybrid: OpenMP decode+ECC, then 1-core infer", "hybrid 2-phase")
                        .replace("C hybrid pipeline: 3 decode+ECC workers + 1 infer core", "hybrid pipeline")
                        .replace("C ", "").replace(" NEON + OpenMP", " + OpenMP"))
    ax.set_xticks(x, [short(r["method"]) for r in crow], rotation=25, ha="right", fontsize=6.5)
    ax.set_ylabel("time (ms)"); ax.set_title("(a) C: stage times and end-to-end")
    ax.set_ylim(0, max(bottom.max(), e.max()) * 1.55)
    ax.legend(fontsize=6.5, loc="upper center", ncol=2); ax.grid(axis="x", visible=False)
    ax = axes[1]
    x = np.arange(len(STAGE_ROWS)); wd = 0.2
    for j, (key, lab, col) in enumerate((("decode_ms", "decode", SC["decode"]), ("ecc_ms", "ECC", SC["ECC"]),
                                         ("infer_ms", "inference", SC["inference"]), ("e2e_ms", "end-to-end", "#333333"))):
        ax.bar(x + (j - 1.5) * wd, [r[key] for r in STAGE_ROWS], wd * 0.95, color=col, label=lab)
    ax.set_yscale("log")
    ax.set_xticks(x, [short(r["method"]) for r in STAGE_ROWS], rotation=25, ha="right", fontsize=6.5)
    ax.set_ylabel("time (ms, log)"); ax.set_title(f"(b) all methods: first datagram -> {NS} predictions")
    ax.legend(fontsize=6.5, ncol=4, loc="upper right"); ax.set_ylim(top=max(r["e2e_ms"] for r in STAGE_ROWS) * 20)
    ax.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig4_stage_breakdown")
print("figures saved in", OUT)
""")

nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for i, (kind, src) in enumerate(cells):
    cell = {"cell_type": kind, "id": f"c{i:02d}", "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    nb["cells"].append(cell)
json.dump(nb, open(f"{HERE}/benchmark_rpi.ipynb", "w"), indent=1)
with open(f"{HERE}/benchmark_rpi_cells.py", "w") as f:
    f.write("# same code as benchmark_rpi.ipynb (generated by make_notebook.py), for running without Jupyter\n")
    f.write('import matplotlib; matplotlib.use("Agg")\n')
    for kind, src in cells:
        if kind == "code":
            f.write("\n" + src + "\n")
print("wrote benchmark_rpi.ipynb and benchmark_rpi_cells.py")
