# Tsetlin Machine projects

<p align="center">
  <img src="tm-ids-rpi-testbed/docs/evidence/ethernet_laptop_to_pi.jpg" width="48%" alt="Laptop with Ethernet cable to a Raspberry Pi">
  <img src="tm-ids-rpi-testbed/docs/evidence/four_pis_and_hub.jpg" width="24%" alt="Four Raspberry Pis and a USB hub">
  <br><sub>The testbed: the laptop (MITM) connected by Ethernet to one sender Pi (left); four Raspberry Pis and a USB hub (right).</sub>
</p>

<p align="center">
  <img src="tm-ids-rpi-testbed/images/testbed_accuracy_vs_K.png" width="49%" alt="Testbed accuracy versus K, no ECC vs Hamming(7,4)">
  <img src="tm-ids-rpi-testbed/images/pi4_latency_per_method.png" width="49%" alt="Pi 4 inference latency per method">
  <br><sub>Left: accuracy of the attacked TM (orange: no ECC, blue: Hamming(7,4)) as the number of corrupted bits K grows. Right: Pi inference latency per method, from pure Python to NEON + OpenMP.</sub>
</p>

| Project | What it is |
|---|---|
| [tm-ids-rpi-testbed](tm-ids-rpi-testbed/) | Distributed Tsetlin Machine intrusion detection on a Raspberry Pi testbed. Three sender Pis, one central Pi, and a laptop that acts as a man-in-the-middle and corrupts one Pi's packets; Hamming(7,4) protection; NEON/OpenMP inference. |
| [compression-ecc-experiments](compression-ecc-experiments/) | Compression and error-correcting-code experiments on the TM rule files. |

The code in both projects was written by an AI assistant under my direction; each README states what is verified and what is still unconfirmed.
