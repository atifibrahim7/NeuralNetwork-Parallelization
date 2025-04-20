NVCC = nvcc
ARCH = -arch=sm_75
CC = gcc
CFLAGS = -O2

.PHONY: all v2 v3 v4 v1 clean

all: v2 v3 v4 v1

v2:
	$(NVCC) $(ARCH) v2.cu -o cuda_v2

v3:
	$(NVCC) $(ARCH) v3.cu -o cuda_v3

v4:
	$(NVCC) $(ARCH) v4.cu -o cuda_v4

nn:
	$(CC) $(CFLAGS) v1.c -o v1

clean:
	rm -f cuda_v2 cuda_v3 cuda_v4 v1
	@echo "Cleaned up generated files."
