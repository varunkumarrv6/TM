# Physical 3-TM testbed. Edit net_config.h first (IP addresses), then on each device:
#   central RPi : make receiver          sending RPi n : make sender NODE=n          laptop : make forwarder
CC     ?= gcc
ARCH   := $(shell uname -m)
ifeq ($(ARCH),aarch64)
  OPT := -O3 -mcpu=native
else
  OPT := -O3 -march=native -msse4.1
endif
CFLAGS ?= -Wall -Wextra

receiver: receiver.c net_config.h dist3_config.h neon_compat.h
	$(CC) $(OPT) $(CFLAGS) -o $@ receiver.c

sender: sender.c net_config.h
	@test -n "$(NODE)" || (echo "usage: make sender NODE=0|1|2"; exit 1)
	$(CC) -O2 $(CFLAGS) -DNODE=$(NODE) -o $@ sender.c

forwarder: forwarder.c net_config.h
	$(CC) -O2 $(CFLAGS) -o $@ forwarder.c

clean:
	rm -f receiver sender forwarder

.PHONY: clean sender
