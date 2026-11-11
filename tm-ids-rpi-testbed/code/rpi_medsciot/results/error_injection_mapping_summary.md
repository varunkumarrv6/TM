# Error injection: mapping the RPi run to the full test set + Monte-Carlo

`forwarder_errors.py <attack> --seed S` picks a random set of K literals (K in [kmin,kmax]). When >=2 of them
land in the SAME Hamming(7,4) codeword, that codeword has a 2+ -bit error that CANNOT be repaired, so the model
is altered and accuracy can drop. `map_to_fulltest.py --attack .. --seed S` regenerates the identical set and
replays it on the full 105,066-flow test set.

Benchmark: full test 95.8350 %  (50-sample subset used by the RPi: 50/50 match the no-attack model).

## Over 60 random seeds per attack (`error_injection_mapping.csv`)

| attack  | collisions? | TM2-only full drop | all-3-TMs full drop |
|---------|-------------|--------------------|---------------------|
| include | none (42 seeds) | +0.0000 pp     | +0.000 pp          |
| include | collision (18)  | -0.1635 pp     | -42.75 pp          |
| exclude | none / collision | +0.0000 pp    | +0.000 pp (exclude only removes conditions -> ~no damage) |
| toggle  | none (42 seeds) | +0.0000 pp     | +0.000 pp          |
| toggle  | collision (18)  | -0.1508 pp     | -46.92 pp          |

No colliding codeword -> Hamming repairs every bit -> accuracy unchanged (this is the k=2 PASS case).
A colliding codeword -> uncorrectable -> accuracy drops.

## Example toggle runs with collisions

| seed | K | colliding cw | TM2-only full drop | TM2-only 50-sample | all-3 full drop | all-3 50-sample |
|------|---|--------------|--------------------|--------------------|-----------------|-----------------|
| 3  | 11 | 1 | -0.0866 pp | 49/50 | -25.74 pp | 32/50 |
| 5  | 12 | 1 | -0.2237 pp | 49/50 | -16.06 pp | 21/50 |
| 11 | 18 | 3 | -0.0514 pp | 49/50 | -93.56 pp | 10/50 |
| 27 | 19 | 2 | -0.0504 pp | 49/50 | -89.33 pp | 11/50 |

## How the 50-sample result maps to the 95.8 % benchmark

* The RPi testbed corrupts ONLY TM2 (1 of 3 TMs; the two direct RPis are clean). Because each TM's class score
  is capped at +-T and then summed, the two clean TMs out-vote the corrupted one, so the TM2-only drop is small
  (~0.05-0.22 pp on the full set; typically 1 of the 50 samples flips -> 49/50).
* You cannot linearly scale "1 of 50 samples" to a % of 95.8 %: 1/50 = 2 % but on the full 105,066 flows the same
  attack changes <0.5 % of predictions. The exact full-set number comes from `map_to_fulltest.py` (offline replay),
  not from the 50-sample count.
* To reproduce Monte-Carlo-scale drops on hardware, all 3 TM streams must pass through the laptop (corrupt all 3),
  i.e. the "all-3-TMs" column -> that matches yesterday's Monte-Carlo.

## Comparable to the Monte-Carlo simulation?

Yes. Each error-injection run is one random literal set; its all-3-TMs full-dataset accuracy falls inside the
Monte-Carlo min/max band for the same K and attack (SEQ), e.g. toggle K~10: our draws land in [2.28, 95.84] %
around the MC mean 76.57 %. Exclude stays ~95.8 % in both, include/toggle collapse the same way.
