#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

#define INPUT_SIZE 784
#define HIDDEN_SIZE 100
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01  
#define EPOCHS 3
#define BATCH_SIZE 64  
#define NUM_CLASSES 10

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
    
    double *d_W1_fp32, *d_W2_fp32;
    __half *d_W1_fp16, *d_W2_fp16;
    double *d_b1, *d_b2;
    
    double *d_batch_input, *d_batch_hidden, *d_batch_output, *d_batch_target;
    __half *d_batch_input_fp16, *d_batch_hidden_fp16, *d_batch_d_output_fp16;
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

__global__ void convert_fp32_to_fp16_kernel(double* fp32, __half* fp16, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        fp16[idx] = __double2half(fp32[idx]);
    }
}

__global__ void batch_forward_hidden_wmma(__half* batch_input_fp16, __half* W1_fp16, double* b1, double* batch_hidden, 
                                         int batch_size, int hidden_size, int input_size) {
    int laneId = threadIdx.x % 32;
    int m_tile = blockIdx.y * 16; // batch index
    int n_tile = blockIdx.x * 16; // hidden index
    
    if (m_tile >= batch_size || n_tile >= hidden_size) return;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fragA;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fragB;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fragC;
    
    wmma::fill_fragment(fragC, 0.0f);
    
    int num_k_tiles = (input_size + 15) / 16;
    for (int k = 0; k < num_k_tiles; k++) {
        int k_tile = k * 16;
        
        if (m_tile < batch_size && k_tile < input_size) {
            wmma::load_matrix_sync(fragA, batch_input_fp16 + m_tile * input_size + k_tile, input_size);
        } else {
            wmma::fill_fragment(fragA, 0.0f);
        }
        
        if (k_tile < input_size && n_tile < hidden_size) {
            wmma::load_matrix_sync(fragB, W1_fp16 + k_tile * hidden_size + n_tile, hidden_size);
        } else {
            wmma::fill_fragment(fragB, 0.0f);
        }
        
        wmma::mma_sync(fragC, fragA, fragB, fragC);
    }
    
    __shared__ float C_tile[16][16];
    wmma::store_matrix_sync(&C_tile[0][0], fragC, 16, wmma::mem_row_major);
    
    if (laneId < 16) {
        int n = n_tile + laneId;
        if (n < hidden_size) {
            float bias = (float)b1[n];
            for (int i = 0; i < 16; i++) {
                C_tile[i][laneId] += bias;
            }
        }
    }
    __syncthreads();
    
    for (int i = 0; i < 16; i++) {
        int m = m_tile + i;
        if (m < batch_size) {
            for (int j = 0; j < 16; j++) {
                int n = n_tile + j;
                if (n < hidden_size) {
                    batch_hidden[m * hidden_size + n] = (double)C_tile[i][j];
                }
            }
        }
    }
}

__global__ void batch_forward_output_wmma(__half* batch_hidden_fp16, __half* W2_fp16, double* b2, double* batch_output, 
                                         int batch_size, int output_size, int hidden_size) {
    int laneId = threadIdx.x % 32;
    int m_tile = blockIdx.y * 16; // batch index
    int n_tile = blockIdx.x * 16; // output index
    
    if (m_tile >= batch_size || n_tile >= output_size) return;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fragA;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fragB;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fragC;
    
    wmma::fill_fragment(fragC, 0.0f);
    
    int num_k_tiles = (hidden_size + 15) / 16;
    for (int k = 0; k < num_k_tiles; k++) {
        int k_tile = k * 16;
        
        if (m_tile < batch_size && k_tile < hidden_size) {
            wmma::load_matrix_sync(fragA, batch_hidden_fp16 + m_tile * hidden_size + k_tile, hidden_size);
        } else {
            wmma::fill_fragment(fragA, 0.0f);
        }
        
        if (k_tile < hidden_size && n_tile < output_size) {
            wmma::load_matrix_sync(fragB, W2_fp16 + k_tile * output_size + n_tile, output_size);
        } else {
            wmma::fill_fragment(fragB, 0.0f);
        }
        
        wmma::mma_sync(fragC, fragA, fragB, fragC);
    }
    
    __shared__ float C_tile[16][16];
    wmma::store_matrix_sync(&C_tile[0][0], fragC, 16, wmma::mem_row_major);
    
    if (laneId < 16) {
        int n = n_tile + laneId;
        if (n < output_size) {
            float bias = (float)b2[n];
            for (int i = 0; i < 16; i++) {
                C_tile[i][laneId] += bias;
            }
        }
    }
    __syncthreads();
    
    for (int i = 0; i < 16; i++) {
        int m = m_tile + i;
        if (m < batch_size) {
            for (int j = 0; j < 16; j++) {
                int n = n_tile + j;
                if (n < output_size) {
                    batch_output[m * output_size + n] = (double)C_tile[i][j];
                }
            }
        }
    }
}

__global__ void batch_compute_hidden_gradient_wmma(__half* batch_d_output_fp16, __half* W2_fp16, double* batch_hidden, 
                                                  double* batch_d_hidden, int batch_size, int hidden_size, int output_size) {
    int laneId = threadIdx.x % 32;
    int m_tile = blockIdx.y * 16; // batch index
    int n_tile = blockIdx.x * 16; // hidden index
    
    if (m_tile >= batch_size || n_tile >= hidden_size) return;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fragA;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fragB;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fragC;
    
    wmma::fill_fragment(fragC, 0.0f);
    
    int num_k_tiles = (output_size + 15) / 16;
    for (int k = 0; k < num_k_tiles; k++) {
        int k_tile = k * 16;
        
        if (m_tile < batch_size && k_tile < output_size) {
            wmma::load_matrix_sync(fragA, batch_d_output_fp16 + m_tile * output_size + k_tile, output_size);
        } else {
            wmma::fill_fragment(fragA, 0.0f);
        }
        
        if (k_tile < output_size && n_tile < hidden_size) {
            wmma::load_matrix_sync(fragB, W2_fp16 + k_tile * hidden_size + n_tile, hidden_size);
        } else {
            wmma::fill_fragment(fragB, 0.0f);
        }
        
        wmma::mma_sync(fragC, fragA, fragB, fragC);
    }
    
    __shared__ float C_tile[16][16];
    wmma::store_matrix_sync(&C_tile[0][0], fragC, 16, wmma::mem_row_major);
    
    for (int i = 0; i < 16; i++) {
        int m = m_tile + i;
        if (m < batch_size) {
            for (int j = 0; j < 16; j++) {
                int n = n_tile + j;
                if (n < hidden_size) {
                    double hidden_val = batch_hidden[m * hidden_size + n];
                    batch_d_hidden[m * hidden_size + n] = (double)C_tile[i][j] * (hidden_val > 0 ? 1.0 : 0.0);
                }
            }
        }
    }
}

__global__ void batch_update_output_weights_wmma(__half* batch_d_output_fp16, __half* batch_hidden_fp16, double* d_W2_fp32, 
                                                int batch_size, int output_size, int hidden_size, double learning_rate) {
    int laneId = threadIdx.x % 32;
    int m_tile = blockIdx.y * 16; // output index
    int n_tile = blockIdx.x * 16; // hidden index
    
    if (m_tile >= output_size || n_tile >= hidden_size) return;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> fragA;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fragB;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fragC;
    
    wmma::fill_fragment(fragC, 0.0f);
    
    int num_k_tiles = (batch_size + 15) / 16;
    for (int k = 0; k < num_k_tiles; k++) {
        int k_tile = k * 16;
        
        if (m_tile < output_size && k_tile < batch_size) {
            wmma::load_matrix_sync(fragA, batch_d_output_fp16 + k_tile * output_size + m_tile, output_size);
        } else {
            wmma::fill_fragment(fragA, 0.0f);
        }
        
        if (k_tile < batch_size && n_tile < hidden_size) {
            wmma::load_matrix_sync(fragB, batch_hidden_fp16 + k_tile * hidden_size + n_tile, hidden_size);
        } else {
            wmma::fill_fragment(fragB, 0.0f);
        }
        
        wmma::mma_sync(fragC, fragA, fragB, fragC);
    }
    
    __shared__ float C_tile[16][16];
    wmma::store_matrix_sync(&C_tile[0][0], fragC, 16, wmma::mem_row_major);
    
    for (int i = 0; i < 16; i++) {
        int m = m_tile + i;
        if (m < output_size) {
            for (int j = 0; j < 16; j++) {
                int n = n_tile + j;
                if (n < hidden_size) {
                    double grad = (double)C_tile[i][j] / batch_size;
                    atomicAdd(&d_W2_fp32[m * hidden_size + n], -learning_rate * grad);
                }
            }
        }
    }
}

__global__ void batch_update_hidden_weights_wmma(__half* batch_d_hidden_fp16, __half* batch_input_fp16, double* d_W1_fp32, 
                                                int batch_size, int hidden_size, int input_size, double learning_rate) {
    int laneId = threadIdx.x % 32;
    int m_tile = blockIdx.y * 16; // hidden index
    int n_tile = blockIdx.x * 16; // input index
    
    if (m_tile >= hidden_size || n_tile >= input_size) return;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> fragA;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fragB;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fragC;
    
    wmma::fill_fragment(fragC, 0.0f);
    
    int num_k_tiles = (batch_size + 15) / 16;
    for (int k = 0; k < num_k_tiles; k++) {
        int k_tile = k * 16;
        
        if (m_tile < hidden_size && k_tile < batch_size) {
            wmma::load_matrix_sync(fragA, batch_d_hidden_fp16 + k_tile * hidden_size + m_tile, hidden_size);
        } else {
            wmma::fill_fragment(fragA, 0.0f);
        }
        
        if (k_tile < batch_size && n_tile < input_size) {
            wmma::load_matrix_sync(fragB, batch_input_fp16 + k_tile * input_size + n_tile, input_size);
        } else {
            wmma::fill_fragment(fragB, 0.0f);
        }
        
        wmma::mma_sync(fragC, fragA, fragB, fragC);
    }
    
    __shared__ float C_tile[16][16];
    wmma::store_matrix_sync(&C_tile[0][0], fragC, 16, wmma::mem_row_major);
    
    for (int i = 0; i < 16; i++) {
        int m = m_tile + i;
        if (m < hidden_size) {
            for (int j = 0; j < 16; j++) {
                int n = n_tile + j;
                if (n < input_size) {
                    double grad = (double)C_tile[i][j] / batch_size;
                    atomicAdd(&d_W1_fp32[m * input_size + n], -learning_rate * grad);
                }
            }
        }
    }
}

__global__ void batch_relu_kernel(double* batch_hidden, int batch_size, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;
    
    if (batch_idx < batch_size && idx < hidden_size) {
        int offset = batch_idx * hidden_size + idx;
        batch_hidden[offset] = (batch_hidden[offset] > 0) ? batch_hidden[offset] : 0;
    }
}

__global__ void batch_softmax_kernel(double* batch_output, int batch_size, int output_size) {
    int batch_idx = blockIdx.x;
    
    if (batch_idx < batch_size) {
        double* output = batch_output + batch_idx * output_size;
        
        double max_val = -INFINITY;
        for (int i = 0; i < output_size; i++) {
            if (output[i] > max_val) max_val = output[i];
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

    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_W1_fp32, HIDDEN_SIZE * INPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_W2_fp32, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_W1_fp16, HIDDEN_SIZE * INPUT_SIZE * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_W2_fp16, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_b1, HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_b2, OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_batch_input, BATCH_SIZE * INPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_batch_input_fp16, BATCH_SIZE * INPUT_SIZE * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_batch_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_batch_hidden_fp16, BATCH_SIZE * HIDDEN_SIZE * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_batch_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_batch_target, BATCH_SIZE * OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_batch_d_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_batch_d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_batch_d_output_fp16, BATCH_SIZE * OUTPUT_SIZE * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_input, INPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_hidden, HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_output, OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_target, OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_d_hidden, HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->d_d_output, OUTPUT_SIZE * sizeof(double)));

    CHECK_CUDA_ERROR(cudaMemcpy(net->d_W1_fp32, flat_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_W2_fp32, flat_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_b1, net->b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_b2, net->b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    
    int blockSize = 256;
    int numBlocks = (HIDDEN_SIZE * INPUT_SIZE + blockSize - 1) / blockSize;
    convert_fp32_to_fp16_kernel<<<numBlocks, blockSize>>>(net->d_W1_fp32, net->d_W1_fp16, HIDDEN_SIZE * INPUT_SIZE);
    numBlocks = (OUTPUT_SIZE * HIDDEN_SIZE + blockSize - 1) / blockSize;
    convert_fp32_to_fp16_kernel<<<numBlocks, blockSize>>>(net->d_W2_fp32, net->d_W2_fp16, OUTPUT_SIZE * HIDDEN_SIZE);
    
    free(flat_W1);
    free(flat_W2);
    
    printf("GPU memory allocation and transfer time: %.3fs\n", get_time(mem_start));
    printf("Batch size: %d\n", BATCH_SIZE);

    return net;
}

void forward(NeuralNetwork* net, double* input, double* hidden, double* output, double* debugTimes) {
    cudaStream_t stream1, stream2;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream1));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream2));

    clock_t transfer_start = clock();
    CHECK_CUDA_ERROR(cudaMemcpyAsync(net->d_input, input, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice, stream1));
    int blockSize = 256;
    int numBlocks = (INPUT_SIZE + blockSize - 1) / blockSize;
    convert_fp32_to_fp16_kernel<<<numBlocks, blockSize, 0, stream1>>>(net->d_input, net->d_batch_input_fp16, INPUT_SIZE);
    debugTimes[0] += get_time(transfer_start); 

    clock_t compute_start = clock();
    
    dim3 gridSize((HIDDEN_SIZE + 15) / 16, 1);
    batch_forward_hidden_wmma<<<gridSize, 32, 0, stream1>>>(
        net->d_batch_input_fp16, net->d_W1_fp16, net->d_b1, net->d_hidden, 1, HIDDEN_SIZE, INPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());

    numBlocks = (HIDDEN_SIZE + blockSize - 1) / blockSize;
    batch_relu_kernel<<<numBlocks, blockSize, 0, stream1>>>(net->d_hidden, 1, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    convert_fp32_to_fp16_kernel<<<numBlocks, blockSize, 0, stream1>>>(net->d_hidden, net->d_batch_hidden_fp16, HIDDEN_SIZE);
    
    dim3 gridSize_output((OUTPUT_SIZE + 15) / 16, 1);
    batch_forward_output_wmma<<<gridSize_output, 32, 0, stream1>>>(
        net->d_batch_hidden_fp16, net->d_W2_fp16, net->d_b2, net->d_output, 1, OUTPUT_SIZE, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    batch_softmax_kernel<<<1, 32, 0, stream1>>>(net->d_output, 1, OUTPUT_SIZE);
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

void backward(NeuralNetwork* net, double* input, double* hidden, double* output, double* target, double* debugTimes, int imageIndex, int numImages, int epoch, int batchSize) {
    clock_t transfer_start = clock();
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_input, input, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_hidden, hidden, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_output, output, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_target, target, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    debugTimes[3] += get_time(transfer_start);
    
    clock_t compute_start = clock();
    
    int blockSize = 256;
    int numBlocks = (OUTPUT_SIZE + blockSize - 1) / blockSize;
    batch_compute_output_gradient_kernel<<<numBlocks, blockSize>>>(net->d_output, net->d_target, net->d_d_output, 1, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    convert_fp32_to_fp16_kernel<<<numBlocks, blockSize>>>(net->d_d_output, net->d_batch_d_output_fp16, OUTPUT_SIZE);
    
    dim3 gridSize_hidden((HIDDEN_SIZE + 15) / 16, 1);
    batch_compute_hidden_gradient_wmma<<<gridSize_hidden, 32>>>(net->d_batch_d_output_fp16, net->d_W2_fp16, net->d_hidden, net->d_d_hidden, 1, HIDDEN_SIZE, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    dim3 gridDim_output((OUTPUT_SIZE + 15) / 16, (HIDDEN_SIZE + 15) / 16);
    batch_update_output_weights_wmma<<<gridDim_output, 32>>>(net->d_batch_d_output_fp16, net->d_batch_hidden_fp16, net->d_W2_fp32, 1, OUTPUT_SIZE, HIDDEN_SIZE, LEARNING_RATE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    numBlocks = (HIDDEN_SIZE + blockSize - 1) / blockSize;
    convert_fp32_to_fp16_kernel<<<numBlocks, blockSize>>>(net->d_d_hidden, net->d_batch_hidden_fp16, HIDDEN_SIZE);
    
    dim3 gridDim_hidden((HIDDEN_SIZE + 15) / 16, (INPUT_SIZE + 15) / 16);
    batch_update_hidden_weights_wmma<<<gridDim_hidden, 32>>>(net->d_batch_hidden_fp16, net->d_batch_input_fp16, net->d_W1_fp32, 1, HIDDEN_SIZE, INPUT_SIZE, LEARNING_RATE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    batch_update_output_bias_kernel<<<numBlocks, blockSize>>>(net->d_b2, net->d_d_output, 1, OUTPUT_SIZE, LEARNING_RATE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    numBlocks = (HIDDEN_SIZE + blockSize - 1) / blockSize;
    batch_update_hidden_bias_kernel<<<numBlocks, blockSize>>>(net->d_b1, net->d_d_hidden, 1, HIDDEN_SIZE, LEARNING_RATE);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    debugTimes[4] += get_time(compute_start);
    
    static double* flat_W1 = NULL;
    static double* flat_W2 = NULL;
    
    bool isEpochEnd = (imageIndex == numImages - 1);
    bool isBatchEnd = ((imageIndex + 1) % batchSize == 0);
    bool shouldSyncWeights = isEpochEnd || (isBatchEnd && epoch == EPOCHS - 1);
    
    if (shouldSyncWeights) {
        transfer_start = clock();
        
        if (flat_W1 == NULL) flat_W1 = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
        if (flat_W2 == NULL) flat_W2 = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
        
        CHECK_CUDA_ERROR(cudaMemcpy(flat_W1, net->d_W1_fp32, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(flat_W2, net->d_W2_fp32, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(net->b1, net->d_b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(net->b2, net->d_b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
        
        unflatten_matrix(flat_W1, net->W1, HIDDEN_SIZE, INPUT_SIZE);
        unflatten_matrix(flat_W2, net->W2, OUTPUT_SIZE, HIDDEN_SIZE);
        
        debugTimes[5] += get_time(transfer_start);
        
        if (isEpochEnd) printf("\nSynchronized weights from GPU to CPU (end of epoch)\n");
    }
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
    
    int blockSize = 256;
    int numBlocks = (actual_batch_size * INPUT_SIZE + blockSize - 1) / blockSize;
    convert_fp32_to_fp16_kernel<<<numBlocks, blockSize, 0, stream>>>(net->d_batch_input, net->d_batch_input_fp16, actual_batch_size * INPUT_SIZE);
    debugTimes[0] += get_time(transfer_start);
    
    clock_t compute_start = clock();
    
    dim3 gridSize((HIDDEN_SIZE + 15) / 16, (actual_batch_size + 15) / 16);
    batch_forward_hidden_wmma<<<gridSize, 32, 0, stream>>>(
        net->d_batch_input_fp16, net->d_W1_fp16, net->d_b1, net->d_batch_hidden,
        actual_batch_size, HIDDEN_SIZE, INPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    dim3 relu_gridSize((HIDDEN_SIZE + blockSize - 1) / blockSize, actual_batch_size);
    batch_relu_kernel<<<relu_gridSize, blockSize, 0, stream>>>(
        net->d_batch_hidden, actual_batch_size, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    numBlocks = (actual_batch_size * HIDDEN_SIZE + blockSize - 1) / blockSize;
    convert_fp32_to_fp16_kernel<<<numBlocks, blockSize, 0, stream>>>(net->d_batch_hidden, net->d_batch_hidden_fp16, actual_batch_size * HIDDEN_SIZE);
    
    dim3 output_gridSize((OUTPUT_SIZE + 15) / 16, (actual_batch_size + 15) / 16);
    batch_forward_output_wmma<<<output_gridSize, 32, 0, stream>>>(
        net->d_batch_hidden_fp16, net->d_W2_fp16, net->d_b2, net->d_batch_output,
        actual_batch_size, OUTPUT_SIZE, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    batch_softmax_kernel<<<actual_batch_size, 32, 0, stream>>>(
        net->d_batch_output, actual_batch_size, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    debugTimes[1] += get_time(compute_start);
    
    transfer_start = clock();
    CHECK_CUDA_ERROR(cudaMemcpyAsync(batch_hidden, net->d_batch_hidden,
                                    actual_batch_size * HIDDEN_SIZE * sizeof(double),
                                    cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(batch_output, net->d_batch_output,
                                    actual_batch_size * OUTPUT_SIZE * sizeof(double),
                                    cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    debugTimes[2] += get_time(transfer_start);
    
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
    
    int blockSize = 256;
    int numBlocks = (actual_batch_size * INPUT_SIZE + blockSize - 1) / blockSize;
    convert_fp32_to_fp16_kernel<<<numBlocks, blockSize, 0, stream>>>(net->d_batch_input, net->d_batch_input_fp16, actual_batch_size * INPUT_SIZE);
    debugTimes[3] += get_time(transfer_start);
    
    clock_t compute_start = clock();
    
    dim3 gridSize_out((OUTPUT_SIZE + blockSize - 1) / blockSize, actual_batch_size);
    batch_compute_output_gradient_kernel<<<gridSize_out, blockSize, 0, stream>>>(
        net->d_batch_output, net->d_batch_target, net->d_batch_d_output,
        actual_batch_size, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    numBlocks = (actual_batch_size * OUTPUT_SIZE + blockSize - 1) / blockSize;
    convert_fp32_to_fp16_kernel<<<numBlocks, blockSize, 0, stream>>>(net->d_batch_d_output, net->d_batch_d_output_fp16, actual_batch_size * OUTPUT_SIZE);
    
    dim3 gridSize_hidden((HIDDEN_SIZE + 15) / 16, (actual_batch_size + 15) / 16);
    batch_compute_hidden_gradient_wmma<<<gridSize_hidden, 32, 0, stream>>>(
        net->d_batch_d_output_fp16, net->d_W2_fp16, net->d_batch_hidden, net->d_batch_d_hidden,
        actual_batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    dim3 grid_weights_out((OUTPUT_SIZE + 15) / 16, (HIDDEN_SIZE + 15) / 16);
    batch_update_output_weights_wmma<<<grid_weights_out, 32, 0, stream>>>(
        net->d_batch_d_output_fp16, net->d_batch_hidden_fp16, net->d_W2_fp32,
        actual_batch_size, OUTPUT_SIZE, HIDDEN_SIZE, LEARNING_RATE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    numBlocks = (actual_batch_size * HIDDEN_SIZE + blockSize - 1) / blockSize;
    convert_fp32_to_fp16_kernel<<<numBlocks, blockSize, 0, stream>>>(net->d_batch_d_hidden, net->d_batch_hidden_fp16, actual_batch_size * HIDDEN_SIZE);
    
    dim3 grid_weights_hidden((HIDDEN_SIZE + 15) / 16, (INPUT_SIZE + 15) / 16);
    batch_update_hidden_weights_wmma<<<grid_weights_hidden, 32, 0, stream>>>(
        net->d_batch_hidden_fp16, net->d_batch_input_fp16, net->d_W1_fp32,
        actual_batch_size, HIDDEN_SIZE, INPUT_SIZE, LEARNING_RATE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    batch_update_output_bias_kernel<<<(OUTPUT_SIZE + 255) / 256, 256, 0, stream>>>(
        net->d_b2, net->d_batch_d_output, actual_batch_size, OUTPUT_SIZE, LEARNING_RATE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    batch_update_hidden_bias_kernel<<<(HIDDEN_SIZE + 255) / 256, 256, 0, stream>>>(
        net->d_b1, net->d_batch_d_hidden, actual_batch_size, HIDDEN_SIZE, LEARNING_RATE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    numBlocks = (HIDDEN_SIZE * INPUT_SIZE + blockSize - 1) / blockSize;
    convert_fp32_to_fp16_kernel<<<numBlocks, blockSize, 0, stream>>>(net->d_W1_fp32, net->d_W1_fp16, HIDDEN_SIZE * INPUT_SIZE);
    numBlocks = (OUTPUT_SIZE * HIDDEN_SIZE + blockSize - 1) / blockSize;
    convert_fp32_to_fp16_kernel<<<numBlocks, blockSize, 0, stream>>>(net->d_W2_fp32, net->d_W2_fp16, OUTPUT_SIZE * HIDDEN_SIZE);
    
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    debugTimes[4] += get_time(compute_start);
    
    static double* flat_W1 = NULL;
    static double* flat_W2 = NULL;
    
    bool isLastBatch = (batchIndex == numBatches - 1);
    bool isLastEpoch = (epoch == EPOCHS - 1);
    
    if (isLastBatch || (isLastEpoch && batchIndex % 10 == 0)) {
        transfer_start = clock();
        
        if (flat_W1 == NULL) flat_W1 = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
        if (flat_W2 == NULL) flat_W2 = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
        
        CHECK_CUDA_ERROR(cudaMemcpy(flat_W1, net->d_W1_fp32, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(flat_W2, net->d_W2_fp32, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(net->b1, net->d_b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(net->b2, net->d_b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
        
        unflatten_matrix(flat_W1, net->W1, HIDDEN_SIZE, INPUT_SIZE);
        unflatten_matrix(flat_W2, net->W2, OUTPUT_SIZE, HIDDEN_SIZE);
        
        debugTimes[5] += get_time(transfer_start);
        
        if (isLastBatch) printf("\nSynchronized weights from GPU to CPU (end of epoch)\n");
    }
    
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}

void train(NeuralNetwork* net, double** images, double** labels, int numImages) {
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
        for (int i = 0; i < 6; i++) debugTimes[i] = 0;
        
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
    CHECK_CUDA_ERROR(cudaFree(net->d_W1_fp32));
    CHECK_CUDA_ERROR(cudaFree(net->d_W2_fp32));
    CHECK_CUDA_ERROR(cudaFree(net->d_W1_fp16));
    CHECK_CUDA_ERROR(cudaFree(net->d_W2_fp16));
    CHECK_CUDA_ERROR(cudaFree(net->d_b1));
    CHECK_CUDA_ERROR(cudaFree(net->d_b2));
    CHECK_CUDA_ERROR(cudaFree(net->d_batch_input));
    CHECK_CUDA_ERROR(cudaFree(net->d_batch_input_fp16));
    CHECK_CUDA_ERROR(cudaFree(net->d_batch_hidden));
    CHECK_CUDA_ERROR(cudaFree(net->d_batch_hidden_fp16));
    CHECK_CUDA_ERROR(cudaFree(net->d_batch_output));
    CHECK_CUDA_ERROR(cudaFree(net->d_batch_target));
    CHECK_CUDA_ERROR(cudaFree(net->d_batch_d_hidden));
    CHECK_CUDA_ERROR(cudaFree(net->d_batch_d_output));
    CHECK_CUDA_ERROR(cudaFree(net->d_batch_d_output_fp16));
    CHECK_CUDA_ERROR(cudaFree(net->d_input));
    CHECK_CUDA_ERROR(cudaFree(net->d_hidden));
    CHECK_CUDA_ERROR(cudaFree(net->d_output));
    CHECK_CUDA_ERROR(cudaFree(net->d_target));
    CHECK_CUDA_ERROR(cudaFree(net->d_d_hidden));
    CHECK_CUDA_ERROR(cudaFree(net->d_d_output));
    
    freeMatrix(net->W1, HIDDEN_SIZE);
    freeMatrix(net->W2, OUTPUT_SIZE);
    free(net->b1);
    free(net->b2);
    free(net);
}

int main() {
    printf(" Neural Network (CUDA Implementation with Tensor Cores)\n\n");
    
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
