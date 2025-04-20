# Makefile

NVCC = nvcc
ARCH = -arch=sm_75

# Targets
all: v2 v3

v2:
	$(NVCC) $(ARCH) v2.cu -o cuda_v2

v3:
	$(NVCC) $(ARCH) v3.cu -o cuda_v3

clean:
	rm -f cuda_v2 cuda_v3
	@echo "Cleaned up generated files."