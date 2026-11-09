# Build of RK3588S (aarch64) NPU verification programs.
# Example:
#   make CROSS=/opt/gcc-arm-11.2/bin/aarch64-none-linux-gnu-gcc LIBRKNN=/path/to/dir_with_librknnrt
CROSS   ?= aarch64-none-linux-gnu-gcc
LIBRKNN ?= .
CFLAGS  ?= -O2 -Wall -Wextra
PROGS   := npu/npu_check npu/npu_detect

all: $(PROGS)

npu/npu_check: npu/npu_check.c
	$(CROSS) $(CFLAGS) -o $@ $< -L$(LIBRKNN) -lrknnrt

npu/npu_detect: npu/npu_detect.c
	$(CROSS) $(CFLAGS) -o $@ $< -L$(LIBRKNN) -lrknnrt

clean:
	rm -f $(PROGS)

.PHONY: all clean
