/*
 * Network of the physical testbed -- the only file with addresses. sender.c, receiver.c, forwarder.c and
 * forwarder.py all read it; no IPs are needed on the command line.
 *
 *   RPi 0 (TM 0)  RPI0_IP 10.42.0.1    ─────────────────────────────────────────────────► CENTRAL_IP:CENTRAL_PORT
 *   RPi 2 (TM 2)  RPI2_IP 10.42.0.59   ─────────────────────────────────────────────────► CENTRAL_IP:CENTRAL_PORT
 *   RPi 1 (TM 1)  RPI1_IP 192.168.50.3 ──► LAPTOP_IP:LAPTOP_PORT (forwarder, attack with K) ──► CENTRAL_IP
 *                                                                          central RPi 10.42.0.145 (receiver)
 * VIA_LAPTOP_NODE = the TM whose RPi is connected to the laptop (that sender sends to the laptop).
 */
#ifndef NET_CONFIG_H
#define NET_CONFIG_H

#ifndef VIA_LAPTOP_NODE
#define VIA_LAPTOP_NODE 1              /* TM 1 goes through the laptop                    */
#endif
#ifndef CENTRAL_IP
#define CENTRAL_IP   "10.42.0.145"     /* central RPi (receiver)                          */
#endif
#ifndef CENTRAL_PORT
#define CENTRAL_PORT 5001
#endif
#ifndef LAPTOP_IP
#define LAPTOP_IP    "192.168.50.2"    /* laptop interface that RPi 1 can reach           */
#endif
#ifndef LAPTOP_PORT
#define LAPTOP_PORT  5001
#endif
#ifndef RPI0_IP
#define RPI0_IP      "10.42.0.1"       /* sending RPi of TM 0 (sends to the central RPi)  */
#endif
#ifndef RPI1_IP
#define RPI1_IP      "192.168.50.3"    /* sending RPi of TM 1 (sends to the LAPTOP)       */
#endif
#ifndef RPI2_IP
#define RPI2_IP      "10.42.0.59"      /* sending RPi of TM 2 (sends to the central RPi)  */
#endif
#ifndef GAP_US
#define GAP_US       2000              /* pause between datagrams; raise it if packets get lost */
#endif

#endif
