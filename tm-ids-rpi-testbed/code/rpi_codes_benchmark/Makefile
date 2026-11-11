# Build all 10 reconstructed TM kernels + the original inter-clause program.
#   Raspberry Pi 4 (aarch64): real <arm_neon.h>, -mcpu=native
#   x86-64 PC               : NEON intrinsics mapped to SSE4.1 via shim/

CC     ?= gcc
ARCH   := $(shell uname -m)

ifeq ($(ARCH),aarch64)
  CFLAGS ?= -O3 -mcpu=native -Wall -Wextra
  ORIG_INC :=
else
  CFLAGS ?= -O3 -march=native -msse4.1 -Wall -Wextra
  ORIG_INC := -Ishim
endif

KERNELS := tm01_normal tm02_packed_bytes tm03_packed_early_exit tm04_packed64 \
           tm05_packed_batch tm06_scalar_neon tm07_persistent_neon \
           tm08_interclause_neon tm09_interclause_omp tm10_final_neon

BIN := bin

all: $(addprefix $(BIN)/,$(KERNELS)) $(BIN)/export_bins $(BIN)/original_interclause

$(BIN):
	mkdir -p $(BIN)

$(BIN)/tm09_interclause_omp: tm09_interclause_omp.c tm_common.h neon_compat.h | $(BIN)
	$(CC) $(CFLAGS) -fopenmp -o $@ $<

$(BIN)/%: %.c tm_common.h neon_compat.h | $(BIN)
	$(CC) $(CFLAGS) -o $@ $<

$(BIN)/original_interclause: original/tm_interclause_neon_single.c | $(BIN)
	$(CC) $(CFLAGS) -Wno-format-extra-args $(ORIG_INC) -o $@ $<

run: all
	./run_all.sh

clean:
	rm -rf $(BIN) original_run

.PHONY: all run clean
