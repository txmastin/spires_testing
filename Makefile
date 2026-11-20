# Builds the C runners and tests in src/bench against a built SPIRES checkout.
#
#   make SPIRES_DIR=/path/to/spires      (default: ~/programming/projects/spiking-reservoir/spires)
#
# clang, not gcc: libspires.a carries LLVM OpenMP symbols and will not link
# against libgomp. Build SPIRES first (`make` in its checkout).

SPIRES_DIR ?= $(HOME)/programming/projects/spiking-reservoir/spires
CC       := clang
CFLAGS   := -O2 -fopenmp -I$(SPIRES_DIR)/include -I$(SPIRES_DIR)/src \
            -I$(SPIRES_DIR)/src/neurons -I$(SPIRES_DIR)/src/synapses
LDLIBS   := $(SPIRES_DIR)/lib/libspires.a -lopenblas -lm

# verify_exact_harness.c is built by verify_exact.py itself.
SRCS := $(filter-out src/bench/verify_exact_harness.c,$(wildcard src/bench/*.c))
BINS := $(SRCS:.c=)

all: $(BINS)

src/bench/%: src/bench/%.c $(SPIRES_DIR)/lib/libspires.a
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

clean:
	rm -f $(BINS) results/.verify_exact_harness

.PHONY: all clean
