# Neural Network Acceleration on GPUs - MNIST Classification

This project demonstrates the acceleration of neural network training for MNIST digit classification using GPU-based computation. The implementation progresses from a baseline CPU version to increasingly optimized GPU versions using CUDA and Tensor Cores.

## Project Overview

The project implements a fully connected neural network for classifying images from the MNIST dataset, with a focus on leveraging GPU acceleration to improve training speed and performance. We explore various GPU optimization techniques and their impact on training time and accuracy.

### Neural Network Architecture
- **Input Layer**: 28x28 grayscale MNIST images flattened into a 784-dimensional vector
- **Hidden Layer**: 100 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax activation for digit classification (0-9)

## Implementation Versions

### V1: Sequential CPU Implementation (Baseline)
- Pure CPU-based implementation with sequential processing
- Manual matrix multiplication using nested loops
- No parallelization or hardware acceleration

### V2: Naive CUDA Implementation
- Basic GPU offloading using CUDA
- Parallelized core operations with CUDA kernels
- Explicit memory management between host and device
- Maintains accuracy but faces performance bottlenecks due to data transfer overhead

### V3: Optimized CUDA Implementation
- Advanced memory management with pinned memory
- Tiled matrix multiplication using shared memory
- Kernel fusion to reduce launch overhead
- CUDA streams for overlapping data transfer and computation
- Optimized batch processing using grid-z
- Achieves ~16x speedup over the baseline implementation

### V4: Tensor Core Implementation
- Utilizes specialized Tensor Core hardware for matrix operations
- Implements FP16 (half-precision) computation
- Leverages cuBLAS with Tensor Core support
- Provides further performance improvements with slight accuracy trade-offs

## Requirements

- CUDA-capable GPU with compute capability 7.5 or higher
- CUDA Toolkit (compatible with your GPU)
- GCC Compiler
- Make utility

## Compilation Instructions

The project includes a Makefile for easy compilation of all versions:

```bash
# Compile all versions
make all

# Compile specific versions
make v1  # Sequential CPU version
make v2  # Naive CUDA implementation
make v3  # Optimized CUDA implementation
make v4  # Tensor Core implementation

# Clean up compiled files
make clean
```

## Execution

After compiling, run each version with:

```bash
# Run the sequential CPU version
./v1

# Run the naive CUDA implementation
./cuda_v2

# Run the optimized CUDA implementation
./cuda_v3

# Run the Tensor Core implementation
./cuda_v4
```

## Key Optimizations

1. **Memory Management**:
   - Pinned memory for faster host-device transfers
   - Shared memory tiling for reduced global memory access
   - Persistent memory allocation across epochs

2. **Computation Optimizations**:
   - Tiled matrix multiplication
   - Kernel fusion for related operations
   - Loop unrolling for improved instruction-level parallelism

3. **Parallelism Enhancements**:
   - CUDA streams for overlapping operations
   - Grid-z based batch processing
   - Optimized thread block dimensions

4. **Precision Adjustments**:
   - FP16 computation with Tensor Cores in V4

## Project Team
- Atif Ibrahim (I221249)
- Muhammad Ali (I220827)
- Hussain Ali Zaidi (I220902)

## Performance Results

The project achieves significant speedups through progressive optimization:
- V1 to V3: ~16x speedup through CUDA optimizations
- Further improvements in V4 with Tensor Cores

Note: While optimizations dramatically improve speed, there may be minor impacts on accuracy, particularly in the first epoch and when using reduced precision (FP16) in the Tensor Core implementation.
