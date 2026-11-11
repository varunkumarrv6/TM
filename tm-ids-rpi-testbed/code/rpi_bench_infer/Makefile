# Build the 10 TM inference kernels (MedSec-25 single TM, datagram stream input).
#   Raspberry Pi 4 (aarch64): real <arm_neon.h>, -mcpu=native
#   x86-64 PC               : NEON intrinsics mapped to SSE4.1 via neon_compat.h (for checking only)

CC     ?= gcc
ARCH   := $(shell uname -m)

ifeq ($(ARCH),aarch64)
  CFLAGS ?= -O3 -mcpu=native -Wall -Wextra
else
  CFLAGS ?= -O3 -march=native -msse4.1 -Wall -Wextra
endif

KERNELS := tm01_normal tm02_packed_bytes tm03_packed_early_exit tm04_packed64 \
           tm05_packed_batch tm06_scalar_neon tm07_persistent_neon \
           tm08_interclause_neon tm09_interclause_omp tm10_final_neon tm11_pipeline
BIN := bin

all: $(addprefix $(BIN)/,$(KERNELS))

$(BIN):
	mkdir -p $(BIN)

$(BIN)/tm09_interclause_omp: tm09_interclause_omp.c tm_common.h neon_compat.h model_dims.h | $(BIN)
	$(CC) $(CFLAGS) -fopenmp -o $@ $<

$(BIN)/tm11_pipeline: tm11_pipeline.c tm_common.h neon_compat.h model_dims.h | $(BIN)
	$(CC) $(CFLAGS) -Wno-unused-function -pthread -fopenmp -o $@ $<

$(BIN)/%: %.c tm_common.h neon_compat.h model_dims.h | $(BIN)
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -rf $(BIN)

.PHONY: all clean
