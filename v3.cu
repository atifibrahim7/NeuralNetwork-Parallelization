#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 100
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01  
#define EPOCHS 3
#define BATCH_SIZE 64  
#define NUM_CLASSES 10  // Digits 0-9

#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

double** allocateMatrix(int rows, int cols) {
    double** mat = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        mat[i] = (double*)malloc(cols * sizeof(double));
    }
    return mat;
}

void freeMatrix(double** mat, int rows) {
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

typedef struct {
    double** W1;
    double** W2;
    double* b1;
    double* b2;
    
    double *d_W1, *d_W2;
    double *d_b1, *d_b2;
    
    double *d_batch_input, *d_batch_hidden, *d_batch_output, *d_batch_target;
    double *d_batch_d_hidden, *d_batch_d_output;
    
    double *d_input, *d_hidden, *d_output, *d_target;
    double *d_d_hidden, *d_d_output;
} NeuralNetwork;

double* flatten_matrix(double** matrix, int rows, int cols) {
    double* flat = (double*)malloc(rows * cols * sizeof(double));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            flat[i * cols + j] = matrix[i][j];
        }
    }
    return flat;
}

void unflatten_matrix(double* flat, double** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = flat[i * cols + j];
        }
    }
}



#define TILE_WIDTH 16

__global__ void forward_hidden_kernel_optimized(double* W1, double* input, double* b1, double* hidden, 
                                              int hidden_size, int input_size) {
    __shared__ double input_shared[TILE_WIDTH];
    __shared__ double bias_shared[TILE_WIDTH];
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;
    
    // Load bias into shared memory 
    if (row < hidden_size) {
        bias_shared[threadIdx.x] = b1[row];
    }
    __syncthreads();
    
    //  dot product in tiles to maximize memory coalescing
    for (int tile = 0; tile < (input_size + TILE_WIDTH - 1) / TILE_WIDTH; tile++) {
        int col = tile * TILE_WIDTH + threadIdx.x;
        if (col < input_size) {
            input_shared[threadIdx.x] = input[col];
        } else {
            input_shared[threadIdx.x] = 0.0;
        }
        __syncthreads();
        
        // partial dot product for this tile
        if (row < hidden_size) {
            for (int k = 0; k < TILE_WIDTH && (tile * TILE_WIDTH + k) < input_size; k++) {
                sum += W1[row * input_size + (tile * TILE_WIDTH + k)] * input_shared[k];
            }
        }
        __syncthreads();
    }
    
    if (row < hidden_size) {
        hidden[row] = bias_shared[threadIdx.x] + sum;
    }
}

// Combined ReLU + matrix multiplication for hidden->output layer
__global__ void forward_output_kernel_optimized(double* W2, double* hidden, double* b2, double* output,
                                              int output_size, int hidden_size) {
    __shared__ double hidden_shared[TILE_WIDTH];
    __shared__ double bias_shared[TILE_WIDTH];
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;
    
    // First apply ReLU to hidden values while loading
    if (threadIdx.x < TILE_WIDTH && threadIdx.x < hidden_size) {
        hidden_shared[threadIdx.x] = (hidden[threadIdx.x] > 0) ? hidden[threadIdx.x] : 0;
    }
    
    // Load bias into shared memory for this thread's neuron
    if (row < output_size) {
        bias_shared[threadIdx.x] = b2[row];
    }
    __syncthreads();
    
    // Compute dot product in tiles
    for (int tile = 0; tile < (hidden_size + TILE_WIDTH - 1) / TILE_WIDTH; tile++) {
        // Load hidden tile into shared memory with ReLU already applied
        int col = tile * TILE_WIDTH + threadIdx.x;
        if (col < hidden_size) {
            hidden_shared[threadIdx.x] = (hidden[col] > 0) ? hidden[col] : 0;
        } else {
            hidden_shared[threadIdx.x] = 0.0;
        }
        __syncthreads();
        
        // Compute partial dot product for this tile
        if (row < output_size) {
            for (int k = 0; k < TILE_WIDTH && (tile * TILE_WIDTH + k) < hidden_size; k++) {
                sum += W2[row * hidden_size + (tile * TILE_WIDTH + k)] * hidden_shared[k];
            }
        }
        __syncthreads();
    }
    
    // Write result
    if (row < output_size) {
        output[row] = bias_shared[threadIdx.x] + sum;
    }
}

__global__ void softmax_kernel_optimized(double* x, int size) {
    __shared__ double data[256]; // Shared memory for the array
    __shared__ double max_val;   // For numerical stability
    __shared__ double sum;       // For normalization
    
    // Step 1: Find maximum value for numerical stability
    if (threadIdx.x == 0) {
        max_val = -INFINITY;
        for (int i = 0; i < size; i++) {
            if (x[i] > max_val) max_val = x[i];
        }
    }
    __syncthreads(); 
    
    // Step 2: Compute exp(x - max)
    int idx = threadIdx.x;
    if (idx < size) {
        data[idx] = exp(x[idx] - max_val);
        x[idx] = data[idx]; // Store back to global memory
    } else {
        data[idx] = 0.0;
    }
    __syncthreads(); 
    
    // Step 3:  sum using parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (idx < stride) {
            data[idx] += data[idx + stride];
        }
        __syncthreads(); 
    }
    
    // Step 4: Normalize 
    if (threadIdx.x == 0) {
        sum = data[0];
        if (sum < 1e-10) sum = 1e-10; 
    }
    //// The reason to syncthreads so that the computed thing  is available to all threads
    __syncthreads(); 
    
    if (idx < size) {
        x[idx] /= sum;
    }
}

__global__ void batch_forward_hidden_kernel(double* W1, double* batch_input, double* b1, double* batch_hidden, 
                                           int batch_size, int hidden_size, int input_size) {
    int batch_idx = blockIdx.z;
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && neuron_idx < hidden_size) {
        double* input = batch_input + batch_idx * input_size;
        double* hidden = batch_hidden + batch_idx * hidden_size;
        
        double sum = b1[neuron_idx];
        
        for (int i = 0; i < input_size; i++) {
            sum += W1[neuron_idx * input_size + i] * input[i];
        }
        
        hidden[neuron_idx] = sum;
    }
    
    __syncthreads();
}

__global__ void batch_relu_kernel(double* batch_hidden, int batch_size, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;
    
    if (batch_idx < batch_size && idx < hidden_size) {
        int offset = batch_idx * hidden_size + idx;
        batch_hidden[offset] = (batch_hidden[offset] > 0) ? batch_hidden[offset] : 0;
    }
    __syncthreads(); 
}

__global__ void batch_forward_output_kernel(double* W2, double* batch_hidden, double* b2, double* batch_output, 
                                           int batch_size, int output_size, int hidden_size) {
    int batch_idx = blockIdx.z;
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && neuron_idx < output_size) {
        double* hidden = batch_hidden + batch_idx * hidden_size;
        double* output = batch_output + batch_idx * output_size;
        
        double sum = b2[neuron_idx];
        
        for (int i = 0; i < hidden_size; i++) {
            double hidden_val = (hidden[i] > 0) ? hidden[i] : 0;
            sum += W2[neuron_idx * hidden_size + i] * hidden_val;
        }
        
        output[neuron_idx] = sum;
    }
    
    __syncthreads();
}

__global__ void batch_softmax_kernel(double* batch_output, int batch_size, int output_size) {
    int batch_idx = blockIdx.x;
    
    if (batch_idx < batch_size) {
        double* output = batch_output + batch_idx * output_size;
        
        double max_val = -INFINITY;
        for (int i = 0; i < output_size; i++) {
            if (output[i] > max_val) max_val = output[i]; // Fixed: was incorrectly comparing with max_val
        }
        
        double sum = 0.0;
        for (int i = 0; i < output_size; i++) {
            output[i] = exp(output[i] - max_val);
            sum += output[i];
        }
        
        for (int i = 0; i < output_size; i++) {
            output[i] /= sum;
        }
    }
}

__global__ void batch_compute_output_gradient_kernel(double* batch_output, double* batch_target, 
                                                    double* batch_d_output, int batch_size, int output_size) {
    int batch_idx = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && idx < output_size) {
        int offset = batch_idx * output_size + idx;
        batch_d_output[offset] = batch_output[offset] - batch_target[offset];
    }
}

__global__ void batch_compute_hidden_gradient_kernel(double* W2, double* batch_d_output, double* batch_hidden, 
                                                   double* batch_d_hidden, int batch_size, int hidden_size, int output_size) {
    int batch_idx = blockIdx.z;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && idx < hidden_size) {
        int hidden_offset = batch_idx * hidden_size + idx;
        int output_offset = batch_idx * output_size;
        
        double gradient_sum = 0.0;
        for (int j = 0; j < output_size; j++) {
            gradient_sum += W2[j * hidden_size + idx] * batch_d_output[output_offset + j];
        }
        
        batch_d_hidden[hidden_offset] = gradient_sum * (batch_hidden[hidden_offset] > 0 ? 1.0 : 0.0);
    }
    
    __syncthreads();
}

__global__ void batch_update_output_weights_kernel(double* W2, double* batch_d_output, double* batch_hidden, 
                                                 int batch_size, int output_size, int hidden_size, double learning_rate) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // output neuron
    int j = blockIdx.y * blockDim.y + threadIdx.y; // hidden neuron
    
    if (i < output_size && j < hidden_size) {
        __shared__ double shared_grad_sum[16][16]; // Matches block dimensions
        
        shared_grad_sum[threadIdx.y][threadIdx.x] = 0.0;
        __syncthreads();
        
        double thread_grad_sum = 0.0;
        for (int b = 0; b < batch_size; b++) {
            thread_grad_sum += batch_d_output[b * output_size + i] * batch_hidden[b * hidden_size + j];
        }
        
        shared_grad_sum[threadIdx.y][threadIdx.x] = thread_grad_sum;
        __syncthreads();
        
        W2[i * hidden_size + j] -= learning_rate * (shared_grad_sum[threadIdx.y][threadIdx.x] / batch_size);
    }
}

__global__ void batch_update_hidden_weights_kernel(double* W1, double* batch_d_hidden, double* batch_input, 
                                                 int batch_size, int hidden_size, int input_size, double learning_rate) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // hidden neuron
    int j = blockIdx.y * blockDim.y + threadIdx.y; // input neuron
    
    if (i < hidden_size && j < input_size) {
        // Used shared memory for accumulating the gradients across the batch
        __shared__ double shared_grad_sum[16][16]; 
        
        shared_grad_sum[threadIdx.y][threadIdx.x] = 0.0;
        __syncthreads();
        
        double thread_grad_sum = 0.0;
        for (int b = 0; b < batch_size; b++) {
            thread_grad_sum += batch_d_hidden[b * hidden_size + i] * batch_input[b * input_size + j];
        }
        
        shared_grad_sum[threadIdx.y][threadIdx.x] = thread_grad_sum;
        __syncthreads();
        
        W1[i * input_size + j] -= learning_rate * (shared_grad_sum[threadIdx.y][threadIdx.x] / batch_size);
    }
}

__global__ void batch_update_output_bias_kernel(double* b2, double* batch_d_output, 
                                              int batch_size, int output_size, double learning_rate) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < output_size) {
        double grad_sum = 0;
        
        for (int b = 0; b < batch_size; b++) {
            grad_sum += batch_d_output[b * output_size + i];
        }
        
        b2[i] -= learning_rate * (grad_sum / batch_size);
    }
}

__global__ void batch_update_hidden_bias_kernel(double* b1, double* batch_d_hidden, 
                                              int batch_size, int hidden_size, double learning_rate) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < hidden_size) {
        double grad_sum = 0;
        
        for (int b = 0; b < batch_size; b++) {
            grad_sum += batch_d_hidden[b * hidden_size + i];
        }
        
        b1[i] -= learning_rate * (grad_sum / batch_size);
    }
}

NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->W1 = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    net->W2 = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    net->b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            net->W1[i][j] = ((double)rand() / RAND_MAX) * 0.01;
        }
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            net->W2[i][j] = ((double)rand() / RAND_MAX) * 0.01;
        }
    }

    clock_t mem_start = clock();
    
    double* flat_W1 = flatten_matrix(net->W1, HIDDEN_SIZE, INPUT_SIZE);
    double* flat_W2 = flatten_matrix(net->W2, OUTPUT_SIZE, HIDDEN_SIZE);

    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_b1, HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_b2, OUTPUT_SIZE * sizeof(double)));

    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_input, INPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_hidden, HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_output, OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_target, OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_d_hidden, HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_d_output, OUTPUT_SIZE * sizeof(double)));
    
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_batch_input, BATCH_SIZE * INPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_batch_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_batch_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_batch_target, BATCH_SIZE * OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_batch_d_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_batch_d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(double)));

    CHECK_CUDA_ERROR(cudaMemcpy(net->d_W1, flat_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_W2, flat_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_b1, net->b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_b2, net->b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    
    free(flat_W1);
    free(flat_W2);
    
    printf("GPU memory allocation and transfer time: %.3fs\n", get_time(mem_start));
    printf("Batch size: %d\n", BATCH_SIZE);

    return net;
}

// Forward pass with optimized kernels and CUDA streams for overlapping
void forward(NeuralNetwork* net, double* input, double* hidden, double* output, double* debugTimes) {
    cudaStream_t stream1, stream2;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream1));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream2));

    clock_t transfer_start = clock();
    CHECK_CUDA_ERROR(cudaMemcpyAsync(net->d_input, input, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice, stream1));
    debugTimes[0] += get_time(transfer_start); 

    clock_t compute_start = clock();
    
    dim3 blockSize(TILE_WIDTH, 1);
    dim3 gridSize((HIDDEN_SIZE + TILE_WIDTH - 1) / TILE_WIDTH, 1);
    forward_hidden_kernel_optimized<<<gridSize, blockSize, 0, stream1>>>(
        net->d_W1, net->d_input, net->d_b1, net->d_hidden, HIDDEN_SIZE, INPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());

    
    dim3 gridSize_output((OUTPUT_SIZE + TILE_WIDTH - 1) / TILE_WIDTH, 1);
    forward_output_kernel_optimized<<<gridSize_output, blockSize, 0, stream1>>>(
        net->d_W2, net->d_hidden, net->d_b2, net->d_output, OUTPUT_SIZE, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    softmax_kernel_optimized<<<1, 256, 0, stream1>>>(net->d_output, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));
    
    debugTimes[1] += get_time(compute_start);
    
    transfer_start = clock();
    CHECK_CUDA_ERROR(cudaMemcpyAsync(hidden, net->d_hidden, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost, stream1));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(output, net->d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost, stream2));
    
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2));
    
    debugTimes[2] += get_time(transfer_start); 
    
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream2));
}


void batch_forward(NeuralNetwork* net, double** batch_images, double* batch_hidden, double* batch_output, 
                  int actual_batch_size, double* debugTimes) {
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    
    double* flat_batch_input = (double*)malloc(actual_batch_size * INPUT_SIZE * sizeof(double));
    for (int b = 0; b < actual_batch_size; b++) {
        for (int i = 0; i < INPUT_SIZE; i++) {
            flat_batch_input[b * INPUT_SIZE + i] = batch_images[b][i];
        }
    }
    
    clock_t transfer_start = clock();
    CHECK_CUDA_ERROR(cudaMemcpyAsync(net->d_batch_input, flat_batch_input, 
                                    actual_batch_size * INPUT_SIZE * sizeof(double), 
                                    cudaMemcpyHostToDevice, stream));
    free(flat_batch_input); 
    debugTimes[0] += get_time(transfer_start); 
    
    clock_t compute_start = clock();
    
    dim3 blockSize(32, 1, 1);
    dim3 gridSize((HIDDEN_SIZE + blockSize.x - 1) / blockSize.x, 1, actual_batch_size);
    batch_forward_hidden_kernel<<<gridSize, blockSize, 0, stream>>>(
        net->d_W1, net->d_batch_input, net->d_b1, net->d_batch_hidden,
        actual_batch_size, HIDDEN_SIZE, INPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    dim3 relu_blockSize(32, 1);
    dim3 relu_gridSize((HIDDEN_SIZE + relu_blockSize.x - 1) / relu_blockSize.x, actual_batch_size);
    batch_relu_kernel<<<relu_gridSize, relu_blockSize, 0, stream>>>(
        net->d_batch_hidden, actual_batch_size, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    dim3 output_gridSize((OUTPUT_SIZE + blockSize.x - 1) / blockSize.x, 1, actual_batch_size);
    batch_forward_output_kernel<<<output_gridSize, blockSize, 0, stream>>>(
        net->d_W2, net->d_batch_hidden, net->d_b2, net->d_batch_output,
        actual_batch_size, OUTPUT_SIZE, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    batch_softmax_kernel<<<actual_batch_size, 32, 0, stream>>>(
        net->d_batch_output, actual_batch_size, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    debugTimes[1] += get_time(compute_start); // Forward computation time
    
    transfer_start = clock();
    CHECK_CUDA_ERROR(cudaMemcpyAsync(batch_hidden, net->d_batch_hidden,
                                    actual_batch_size * HIDDEN_SIZE * sizeof(double),
                                    cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(batch_output, net->d_batch_output,
                                    actual_batch_size * OUTPUT_SIZE * sizeof(double),
                                    cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    debugTimes[2] += get_time(transfer_start); // Output transfer time
    
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}

void batch_backward(NeuralNetwork* net, double** batch_images, double* batch_hidden, double* batch_output,
                   double** batch_labels, int actual_batch_size, double* debugTimes, 
                   int batchIndex, int numBatches, int epoch) {
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    
    double* flat_batch_input = (double*)malloc(actual_batch_size * INPUT_SIZE * sizeof(double));
    double* flat_batch_target = (double*)malloc(actual_batch_size * OUTPUT_SIZE * sizeof(double));
    
    for (int b = 0; b < actual_batch_size; b++) {
        for (int i = 0; i < INPUT_SIZE; i++) {
            flat_batch_input[b * INPUT_SIZE + i] = batch_images[b][i];
        }
        
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            flat_batch_target[b * OUTPUT_SIZE + i] = batch_labels[b][i];
        }
    }
    
    clock_t transfer_start = clock();
    CHECK_CUDA_ERROR(cudaMemcpyAsync(net->d_batch_input, flat_batch_input,
                                    actual_batch_size * INPUT_SIZE * sizeof(double),
                                    cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(net->d_batch_target, flat_batch_target,
                                    actual_batch_size * OUTPUT_SIZE * sizeof(double),
                                    cudaMemcpyHostToDevice, stream));
    
    free(flat_batch_input);
    free(flat_batch_target);
    debugTimes[3] += get_time(transfer_start); 
    
    clock_t compute_start = clock();
    
    dim3 blockSize(32);
    dim3 gridSize_out((OUTPUT_SIZE + blockSize.x - 1) / blockSize.x, actual_batch_size);
    batch_compute_output_gradient_kernel<<<gridSize_out, blockSize, 0, stream>>>(
        net->d_batch_output, net->d_batch_target, net->d_batch_d_output,
        actual_batch_size, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    dim3 gridSize_hidden((HIDDEN_SIZE + blockSize.x - 1) / blockSize.x, 1, actual_batch_size);
    batch_compute_hidden_gradient_kernel<<<gridSize_hidden, blockSize, 0, stream>>>(
        net->d_W2, net->d_batch_d_output, net->d_batch_hidden, net->d_batch_d_hidden,
        actual_batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    dim3 block_weights(16, 16);
    dim3 grid_weights_out((OUTPUT_SIZE + block_weights.x - 1) / block_weights.x,
                        (HIDDEN_SIZE + block_weights.y - 1) / block_weights.y);
    batch_update_output_weights_kernel<<<grid_weights_out, block_weights, 0, stream>>>(
        net->d_W2, net->d_batch_d_output, net->d_batch_hidden,
        actual_batch_size, OUTPUT_SIZE, HIDDEN_SIZE, LEARNING_RATE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    dim3 grid_weights_hidden((HIDDEN_SIZE + block_weights.x - 1) / block_weights.x,
                           (INPUT_SIZE + block_weights.y - 1) / block_weights.y);
    batch_update_hidden_weights_kernel<<<grid_weights_hidden, block_weights, 0, stream>>>(
        net->d_W1, net->d_batch_d_hidden, net->d_batch_input,
        actual_batch_size, HIDDEN_SIZE, INPUT_SIZE, LEARNING_RATE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    batch_update_output_bias_kernel<<<(OUTPUT_SIZE + 255) / 256, 256, 0, stream>>>(
        net->d_b2, net->d_batch_d_output, actual_batch_size, OUTPUT_SIZE, LEARNING_RATE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    batch_update_hidden_bias_kernel<<<(HIDDEN_SIZE + 255) / 256, 256, 0, stream>>>(
        net->d_b1, net->d_batch_d_hidden, actual_batch_size, HIDDEN_SIZE, LEARNING_RATE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    debugTimes[4] += get_time(compute_start); // Backward computation time
    
    static double* flat_W1 = NULL;
    static double* flat_W2 = NULL;
    
    bool isLastBatch = (batchIndex == numBatches - 1);
    bool isLastEpoch = (epoch == EPOCHS - 1);
    
    if (isLastBatch || (isLastEpoch && batchIndex % 10 == 0)) {
        transfer_start = clock();
        
        if (flat_W1 == NULL) {
            flat_W1 = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
        }
        if (flat_W2 == NULL) {
            flat_W2 = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
        }
        
        CHECK_CUDA_ERROR(cudaMemcpy(flat_W1, net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(flat_W2, net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(net->b1, net->d_b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(net->b2, net->d_b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
        
        unflatten_matrix(flat_W1, net->W1, HIDDEN_SIZE, INPUT_SIZE);
        unflatten_matrix(flat_W2, net->W2, OUTPUT_SIZE, HIDDEN_SIZE);
        
        debugTimes[5] += get_time(transfer_start); // Weights transfer time
        
        if (isLastBatch) {
            printf("\nSynchronized weights from GPU to CPU (end of epoch)\n");
        }
    }
    
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}

void train(NeuralNetwork* net, double** images, double** labels, int numImages) {
    // Debug timing information
    // [0] = input transfer, [1] = forward computation, [2] = output transfer
    // [3] = backward transfer, [4] = backward computation, [5] = weights transfer
    double debugTimes[6] = {0};
    const char* debugTimesLabels[] = {
        "Host->Device Transfer Time", 
        "Forward Computation Time",
        "Device->Host Transfer Time",
        "Backward Transfer Time",
        "Backward Computation Time",
        "Weights Transfer Time"
    };
    
    clock_t total_start = clock();
    printf("====================================================\n");
    printf("Starting training on GPU with batch processing...\n");
    printf("Batch size: %d\n", BATCH_SIZE);
    printf("Total samples: %d\n", numImages);
    
    int numBatches = (numImages + BATCH_SIZE - 1) / BATCH_SIZE;
    printf("Number of batches per epoch: %d\n", numBatches);
    printf("====================================================\n");
    
    double* batch_hidden = (double*)malloc(BATCH_SIZE * HIDDEN_SIZE * sizeof(double));
    double* batch_output = (double*)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(double));
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int i = 0; i < 6; i++) {
            debugTimes[i] = 0;
        }
        
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;
        int processed = 0;

        for (int batch = 0; batch < numBatches; batch++) {
            int start_idx = batch * BATCH_SIZE;
            int actual_batch_size = (start_idx + BATCH_SIZE <= numImages) ? BATCH_SIZE : (numImages - start_idx);
            
            double** batch_images = &images[start_idx];
            double** batch_labels = &labels[start_idx];
            
            batch_forward(net, batch_images, batch_hidden, batch_output, actual_batch_size, debugTimes);
            batch_backward(net, batch_images, batch_hidden, batch_output, batch_labels, 
                          actual_batch_size, debugTimes, batch, numBatches, epoch);
            
            for (int b = 0; b < actual_batch_size; b++) {
                double* output = &batch_output[b * OUTPUT_SIZE];
                double* label = batch_labels[b];
                
                for (int k = 0; k < OUTPUT_SIZE; k++) {
                    double safe_output = output[k] > 1e-10 ? output[k] : 1e-10;
                    loss -= label[k] * log(safe_output);
                }
                
                int pred = 0, actual = 0;
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    if (output[j] > output[pred]) pred = j;
                    if (label[j] > label[actual]) actual = j;
                }
                if (pred == actual) correct++;
            }
            
            processed += actual_batch_size;
            
            if ((batch + 1) % 100 == 0 || batch == numBatches - 1) {
                printf("\rEpoch %d: Processed %d/%d samples (%.1f%%)...", 
                      epoch + 1, processed, numImages, (processed * 100.0) / numImages);
                fflush(stdout);
            }
        }

        double epoch_time = get_time(epoch_start);
        double accuracy = (correct / (double)numImages) * 100;
        
        printf("\n\n====================================================\n");
        printf("Epoch %d/%d Summary:\n", epoch + 1, EPOCHS);
        printf("----------------------------------------------------\n");
        printf("Loss: %.4f\n", loss / numImages);
        printf("Train Accuracy: %.2f%%\n", accuracy);
        printf("Epoch Time: %.3fs\n", epoch_time);
        
        printf("\nPerformance Breakdown:\n");
        printf("----------------------------------------------------\n");
        for (int i = 0; i < 6; i++) {
            printf("  %-25s: %.6fs (%.2f%%)\n", 
                   debugTimesLabels[i], 
                   debugTimes[i], 
                   (debugTimes[i] / epoch_time) * 100);
        }
        
        double total_transfer_time = debugTimes[0] + debugTimes[2] + debugTimes[3] + debugTimes[5];
        double total_compute_time = debugTimes[1] + debugTimes[4];
        
        printf("\nTime Distribution:\n");
        printf("  Memory Transfer: %.6fs (%.2f%%)\n", 
               total_transfer_time, 
               (total_transfer_time / epoch_time) * 100);
        printf("  GPU Computation: %.6fs (%.2f%%)\n", 
               total_compute_time, 
               (total_compute_time / epoch_time) * 100);
        printf("====================================================\n");
    }
    
    printf("\nTotal training time: %.3fs\n", get_time(total_start));
    printf("====================================================\n\n");
    
    free(batch_hidden);
    free(batch_output);
}

void evaluate(NeuralNetwork* net, double** images, double** labels, int numImages) {
    double debugTimes[6] = {0}; 
    int correct = 0;
    clock_t eval_start = clock();
    
    printf("====================================================\n");
    printf("Evaluating model on test data...\n");
    
    for (int i = 0; i < numImages; i++) {
        double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
        forward(net, images[i], hidden, output, debugTimes);
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred]) pred = j;
            if (labels[i][j] > labels[i][actual]) actual = j;
        }
        if (pred == actual) correct++;
    }
    
    double accuracy = (correct / (double)numImages) * 100;
    double eval_time = get_time(eval_start);
    
    printf("----------------------------------------------------\n");
    printf("Test Accuracy: %.2f%%\n", accuracy);
    printf("Evaluation Time: %.3fs\n", eval_time);
    printf("====================================================\n");
}

double** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET);
    double** images = allocateMatrix(numImages, INPUT_SIZE);
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;

            if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
                fprintf(stderr, "Error: Failed to read pixel\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }

            images[i][j] = pixel / 255.0;
        }
    }
    fclose(file);
    return images;
}

double** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    double** labels = allocateMatrix(numLabels, OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Error: Failed to read label\n");
            fclose(file);
            exit(EXIT_FAILURE);
        }

        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i][j] = (j == label) ? 1.0 : 0.0;
        }
    }
    fclose(file);
    return labels;
}

void freeNetwork(NeuralNetwork* net) {
    CHECK_CUDA_ERROR(cudaFree(net->d_b1));
    CHECK_CUDA_ERROR(cudaFree(net->d_W1));
    CHECK_CUDA_ERROR(cudaFree(net->d_W2));
    CHECK_CUDA_ERROR(cudaFree(net->d_b2));
    CHECK_CUDA_ERROR(cudaFree(net->d_hidden));
    CHECK_CUDA_ERROR(cudaFree(net->d_input));
    CHECK_CUDA_ERROR(cudaFree(net->d_output));
    CHECK_CUDA_ERROR(cudaFree(net->d_d_output));
    CHECK_CUDA_ERROR(cudaFree(net->d_d_hidden));
    CHECK_CUDA_ERROR(cudaFree(net->d_target));
    
    freeMatrix(net->W2, OUTPUT_SIZE);
    freeMatrix(net->W1, HIDDEN_SIZE);
    free(net->b1);
    free(net->b2);
    free(net);
}

int main() {
    printf(" Neural Network (CUDA Implementation)\n\n");
    
    cudaDeviceProp prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));
    printf("Using GPU: %s\n", prop.name);

    double** train_labels = loadMNISTLabels("data/train-labels.idx1-ubyte", 60000);
    double** train_images = loadMNISTImages("data/train-images.idx3-ubyte", 60000);
    double** test_labels = loadMNISTLabels("data/t10k-labels.idx1-ubyte", 10000);
    double** test_images = loadMNISTImages("data/t10k-images.idx3-ubyte", 10000);

    NeuralNetwork* network = createNetwork();
    train(network, train_images, train_labels, 60000);
    evaluate(network, test_images, test_labels, 10000);

    freeNetwork(network);
    
    freeMatrix(train_images, 60000);
    freeMatrix(train_labels, 60000);
    freeMatrix(test_images, 10000);
    freeMatrix(test_labels, 10000);
    
    return 0;
}