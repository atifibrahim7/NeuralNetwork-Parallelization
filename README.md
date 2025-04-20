# CUDA-Accelerated MNIST Classifier

This project implements a neural network for MNIST digit classification with four versions:

- **V1:** Serial CPU implementation  
- **V2:** Naive GPU implementation using CUDA  
- **V3:** Optimized GPU implementation (shared memory, tiling, kernel fusion, streams)  
- **V4:** Tensor Core accelerated implementation  

---

## 🔧 How to Build and Run

1. **Navigate to the source directory:**

   ```bash
   cd src
Build the project using the Makefile:

Terminal Command:
make

This will generate the following binaries:

v1 → V1 (serial CPU)

cuda_v2 → V2 (naive CUDA)

cuda_v3 → V3 (optimized CUDA)

cuda_v4 → V4 (tensor core CUDA)

Run any version as needed:

bash
Copy
Edit
./v1            # Serial CPU version
./cuda_v2       # Naive CUDA version
./cuda_v3       # Optimized CUDA version
./cuda_v4       # Tensor Core version

Dataset Information
The MNIST dataset used in this project is not included in the repository due to its size.

Please refer to info.txt in the root directory for instructions on how to download and place the dataset files.

