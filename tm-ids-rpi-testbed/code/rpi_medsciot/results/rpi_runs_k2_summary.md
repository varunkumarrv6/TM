# RPi testbed runs — k=2 targeted MITM, Hamming(7,4) SEQ

Topology: RPi .1 + RPi .82 send directly to central 10.42.0.59; RPi .3 sends via the
laptop 192.168.50.2 (MITM forwarder) which corrupts TM2 and routes to central.
Benchmark on the 50 test flows: decoded == reference 50/50, accuracy 86.0 % (= deployed 3-TM model, no attack).

| attack  | TM2 codewords corrected | decoded == reference | accuracy | result |
|---------|-------------------------|----------------------|----------|--------|
| include | 8842                    | 50 / 50              | 86.0 %   | PASS   |
| exclude | 1766                    | 50 / 50              | 86.0 %   | PASS   |
| toggle  | 10608                   | 50 / 50              | 86.0 %   | PASS   |

k=2 => the two attacked literals never share a Hamming codeword, so every flipped bit is a single-bit
error that Hamming(7,4) repairs. The model is restored exactly and accuracy is unchanged (PASS).
The number of corrected codewords differs by attack because include only flips 0->1 bits, exclude only
1->0 bits, toggle flips all target bits.

Decode+inference on the central RPi (one core): ~20 ms decode + ~12 ms inference for all 1333 packets.
