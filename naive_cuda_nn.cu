#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 10
#define BATCH_SIZE 64
#define NUM_CLASSES 10  // Digits 0-9
#define BLOCK_SIZE 256  // CUDA block size

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

// Neural network structure
typedef struct {
    // Host memory
    double** W1_host;
    double** W2_host;
    double* b1_host;
    double* b2_host;
    
    // Device memory (flat arrays for GPU)
    double* W1_device;
    double* W2_device;
    double* b1_device;
    double* b2_device;
} NeuralNetwork;

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

double* flattenMatrix(double** matrix, int rows, int cols) {
    double* flat = (double*)malloc(rows * cols * sizeof(double));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            flat[i * cols + j] = matrix[i][j];
        }
    }
    return flat;
}

// Convert flattened 1D array to 2D matrix
void unflattenMatrix(double* flat, double** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = flat[i * cols + j];
        }
    }
}

// CUDA kernel for matrix multiplication: C = A * B
__global__ void matrixMultiplyKernel(double* A, double* B, double* C, int A_rows, int A_cols, int B_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < A_rows && col < B_cols) {
        double sum = 0.0;
        for (int i = 0; i < A_cols; i++) {
            sum += A[row * A_cols + i] * B[i * B_cols + col];
        }
        C[row * B_cols + col] = sum;
    }
}

// CUDA kernel for adding biases: A = A + bias
__global__ void addBiasKernel(double* A, double* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < rows * cols) {
        int row = idx / cols;
        A[idx] += bias[row];
    }
}

// CUDA kernel for ReLU activation: A = max(0, A)
__global__ void reluKernel(double* A, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        A[idx] = (A[idx] > 0) ? A[idx] : 0;
    }
}

// CUDA kernel for softmax activation
__global__ void softmaxKernel(double* A, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        // Find max value in row
        double max_val = A[row * cols];
        for (int i = 1; i < cols; i++) {
            if (A[row * cols + i] > max_val) {
                max_val = A[row * cols + i];
            }
        }
        
        // Compute exp(x - max) for numerical stability
        double sum = 0.0;
        for (int i = 0; i < cols; i++) {
            // Clip very negative values to avoid underflow
            double val = A[row * cols + i] - max_val;
            if (val > -708.0) { // log(DBL_MIN) is around -708
                A[row * cols + i] = exp(val);
            } else {
                A[row * cols + i] = 0.0;
            }
            sum += A[row * cols + i];
        }
        
        // Normalize with epsilon for underflow protection
        const double eps = 1e-15;
        for (int i = 0; i < cols; i++) {
            A[row * cols + i] /= (sum + eps);
        }
    }
}

// CUDA kernel for computing output layer gradient
__global__ void outputGradientKernel(double* output, double* target, double* gradient, int batchSize, int outputSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batchSize * outputSize) {
        gradient[idx] = output[idx] - target[idx];
    }
}

// CUDA kernel for computing hidden layer gradient
__global__ void hiddenGradientKernel(double* d_output, double* W2, double* hidden, double* d_hidden, 
                                    int batchSize, int hiddenSize, int outputSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batchSize * hiddenSize) {
        int batch = idx / hiddenSize;
        int hiddenIdx = idx % hiddenSize;
        
        double sum = 0.0;
        for (int j = 0; j < outputSize; j++) {
            sum += W2[j * hiddenSize + hiddenIdx] * d_output[batch * outputSize + j];
        }
        
        // Apply ReLU derivative
        d_hidden[idx] = sum * (hidden[idx] > 0 ? 1.0 : 0.0);
    }
}

// CUDA kernel for updating weights
__global__ void updateWeightsKernel(double* weights, double* input, double* gradient, 
                                   int batchSize, int inputDim, int outputDim, double learningRate) {
    int outIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int inIdx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (outIdx < outputDim && inIdx < inputDim) {
        double delta = 0.0;
        for (int b = 0; b < batchSize; b++) {
            delta += gradient[b * outputDim + outIdx] * input[b * inputDim + inIdx];
        }
        
        weights[outIdx * inputDim + inIdx] -= learningRate * delta / batchSize;
    }
}

// CUDA kernel for updating biases
__global__ void updateBiasesKernel(double* biases, double* gradient, 
                                  int batchSize, int outputDim, double learningRate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < outputDim) {
        double delta = 0.0;
        for (int b = 0; b < batchSize; b++) {
            delta += gradient[b * outputDim + idx];
        }
        
        biases[idx] -= learningRate * delta / batchSize;
    }
}

NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    
    net->W1_host = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    net->W2_host = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    net->b1_host = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->b2_host = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1_host[i][j] = ((double)rand() / RAND_MAX) * 0.01;

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2_host[i][j] = ((double)rand() / RAND_MAX) * 0.01;
    
    double* W1_flat = flattenMatrix(net->W1_host, HIDDEN_SIZE, INPUT_SIZE);
    double* W2_flat = flattenMatrix(net->W2_host, OUTPUT_SIZE, HIDDEN_SIZE);
    
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->W1_device, HIDDEN_SIZE * INPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->W2_device, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->b1_device, HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&net->b2_device, OUTPUT_SIZE * sizeof(double)));
    
    CHECK_CUDA_ERROR(cudaMemcpy(net->W1_device, W1_flat, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->W2_device, W2_flat, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->b1_device, net->b1_host, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->b2_device, net->b2_host, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    
    free(W1_flat);
    free(W2_flat);
    
    return net;
}

// Read MNIST dataset
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

// Forward pass on GPU for a batch of images
void forwardBatch(NeuralNetwork* net, double* batch_inputs, double* batch_hidden, double* batch_output, int batchSize) {
    dim3 blockDim(16, 16);
    dim3 gridDim1((INPUT_SIZE + blockDim.x - 1) / blockDim.x, (batchSize + blockDim.y - 1) / blockDim.y);
    dim3 gridDim2((HIDDEN_SIZE + blockDim.x - 1) / blockDim.x, (batchSize + blockDim.y - 1) / blockDim.y);
    dim3 gridDim3((OUTPUT_SIZE + blockDim.x - 1) / blockDim.x, (batchSize + blockDim.y - 1) / blockDim.y);
    
    double *d_batch_inputs, *d_batch_hidden, *d_batch_output;
    
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_batch_inputs, batchSize * INPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_batch_hidden, batchSize * HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_batch_output, batchSize * OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemset(d_batch_hidden, 0, batchSize * HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemset(d_batch_output, 0, batchSize * OUTPUT_SIZE * sizeof(double)));
    
    CHECK_CUDA_ERROR(cudaMemcpy(d_batch_inputs, batch_inputs, batchSize * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    
    // ! Hidden layer computation
    // For each sample in batch, :  hidden = W1 * input + b1
    for (int b = 0; b < batchSize; b++) {
        // Matrix multiplication: hidden[b] = W1 * input[b]
        matrixMultiplyKernel<<<gridDim2, blockDim>>>(
            net->W1_device,
            d_batch_inputs + b * INPUT_SIZE,
            d_batch_hidden + b * HIDDEN_SIZE,
            HIDDEN_SIZE, INPUT_SIZE, 1
        );
        
        // Add bias
        addBiasKernel<<<(HIDDEN_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            d_batch_hidden + b * HIDDEN_SIZE,
            net->b1_device,
            1, HIDDEN_SIZE
        );
    }
    
    // Apply ReLU  to hidden layer
    reluKernel<<<(batchSize * HIDDEN_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        d_batch_hidden, batchSize * HIDDEN_SIZE
    );
    
    // Output layer computation
    // For each sample in batch :  output = W2 * hidden + b2
    for (int b = 0; b < batchSize; b++) {
        // Matrix multiplication: output[b] = W2 * hidden[b]
        matrixMultiplyKernel<<<gridDim3, blockDim>>>(
            net->W2_device,
            d_batch_hidden + b * HIDDEN_SIZE,
            d_batch_output + b * OUTPUT_SIZE,
            OUTPUT_SIZE, HIDDEN_SIZE, 1
        );
        
        // Add bias
        addBiasKernel<<<(OUTPUT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            d_batch_output + b * OUTPUT_SIZE,
            net->b2_device,
            1, OUTPUT_SIZE
        );
    }
    
    // softmax to output layer
    softmaxKernel<<<(batchSize + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        d_batch_output, batchSize, OUTPUT_SIZE
    );
    
    CHECK_CUDA_ERROR(cudaMemcpy(batch_hidden, d_batch_hidden, batchSize * HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(batch_output, d_batch_output, batchSize * OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    
    cudaFree(d_batch_inputs);
    cudaFree(d_batch_hidden);
    cudaFree(d_batch_output);
}

// Backward pass on GPU (per batch)
void backwardBatch(NeuralNetwork* net, double* batch_inputs, double* batch_hidden, double* batch_output, double* batch_targets, int batchSize) {
    double *d_batch_inputs, *d_batch_hidden, *d_batch_output, *d_batch_targets;
    double *d_output_grad, *d_hidden_grad;
    
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_batch_inputs, batchSize * INPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_batch_hidden, batchSize * HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_batch_output, batchSize * OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_batch_targets, batchSize * OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output_grad, batchSize * OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_hidden_grad, batchSize * HIDDEN_SIZE * sizeof(double)));
    
    CHECK_CUDA_ERROR(cudaMemcpy(d_batch_inputs, batch_inputs, batchSize * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_batch_hidden, batch_hidden, batchSize * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_batch_output, batch_output, batchSize * OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_batch_targets, batch_targets, batchSize * OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    
    outputGradientKernel<<<(batchSize * OUTPUT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        d_batch_output, d_batch_targets, d_output_grad, batchSize, OUTPUT_SIZE
    );
    
    hiddenGradientKernel<<<(batchSize * HIDDEN_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        d_output_grad, net->W2_device, d_batch_hidden, d_hidden_grad, batchSize, HIDDEN_SIZE, OUTPUT_SIZE
    );
    
    // Update weights and biases
    // For W2: output_dim x hidden_dim
    dim3 blockDim(16, 16);
    dim3 gridDimW2((OUTPUT_SIZE + blockDim.x - 1) / blockDim.x, (HIDDEN_SIZE + blockDim.y - 1) / blockDim.y);
    updateWeightsKernel<<<gridDimW2, blockDim>>>(
        net->W2_device, d_batch_hidden, d_output_grad, batchSize, HIDDEN_SIZE, OUTPUT_SIZE, LEARNING_RATE
    );
    
    // For W1: hidden_dim x input_dim
    dim3 gridDimW1((HIDDEN_SIZE + blockDim.x - 1) / blockDim.x, (INPUT_SIZE + blockDim.y - 1) / blockDim.y);
    updateWeightsKernel<<<gridDimW1, blockDim>>>(
        net->W1_device, d_batch_inputs, d_hidden_grad, batchSize, INPUT_SIZE, HIDDEN_SIZE, LEARNING_RATE
    );
    
    // Update biases
    updateBiasesKernel<<<(OUTPUT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        net->b2_device, d_output_grad, batchSize, OUTPUT_SIZE, LEARNING_RATE
    );
    
    updateBiasesKernel<<<(HIDDEN_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        net->b1_device, d_hidden_grad, batchSize, HIDDEN_SIZE, LEARNING_RATE
    );
    
    cudaFree(d_batch_inputs);
    cudaFree(d_batch_hidden);
    cudaFree(d_batch_output);
    cudaFree(d_batch_targets);
    cudaFree(d_output_grad);
    cudaFree(d_hidden_grad);
}

void train(NeuralNetwork* net, double** images, double** labels, int numImages) {
    clock_t total_start = clock();
    
    double* images_flat = flattenMatrix(images, numImages, INPUT_SIZE);
    double* labels_flat = flattenMatrix(labels, numImages, OUTPUT_SIZE);
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;
        
        for (int batch_start = 0; batch_start < numImages; batch_start += BATCH_SIZE) {
            int actual_batch_size = (batch_start + BATCH_SIZE <= numImages) ? BATCH_SIZE : (numImages - batch_start);
            
            double* batch_hidden = (double*)malloc(actual_batch_size * HIDDEN_SIZE * sizeof(double));
            double* batch_output = (double*)malloc(actual_batch_size * OUTPUT_SIZE * sizeof(double));
            
            forwardBatch(net, images_flat + batch_start * INPUT_SIZE, 
                         batch_hidden, batch_output, actual_batch_size);
            
            backwardBatch(net, images_flat + batch_start * INPUT_SIZE, 
                          batch_hidden, batch_output, 
                          labels_flat + batch_start * OUTPUT_SIZE, actual_batch_size);
            
            for (int b = 0; b < actual_batch_size; b++) {
                for (int k = 0; k < OUTPUT_SIZE; k++) {
                    if (labels_flat[(batch_start + b) * OUTPUT_SIZE + k] > 0) {
                        loss -= log(batch_output[b * OUTPUT_SIZE + k]);
                    }
                }
                
                int pred = 0, actual = 0;
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    if (batch_output[b * OUTPUT_SIZE + j] > batch_output[b * OUTPUT_SIZE + pred]) pred = j;
                    if (labels_flat[(batch_start + b) * OUTPUT_SIZE + j] > labels_flat[(batch_start + b) * OUTPUT_SIZE + actual]) actual = j;
                }
                if (pred == actual) correct++;
            }
            
            free(batch_hidden);
            free(batch_output);
        }
        
        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
    }
    
    double* W1_flat = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    double* W2_flat = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    
    CHECK_CUDA_ERROR(cudaMemcpy(W1_flat, net->W1_device, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(W2_flat, net->W2_device, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(net->b1_host, net->b1_device, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(net->b2_host, net->b2_device, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    
    unflattenMatrix(W1_flat, net->W1_host, HIDDEN_SIZE, INPUT_SIZE);
    unflattenMatrix(W2_flat, net->W2_host, OUTPUT_SIZE, HIDDEN_SIZE);
    
    free(images_flat);
    free(labels_flat);
    free(W1_flat);
    free(W2_flat);
    
    printf("Total training time: %.3fs\n", get_time(total_start));
}

void evaluate(NeuralNetwork* net, double** images, double** labels, int numImages) {
    double* images_flat = flattenMatrix(images, numImages, INPUT_SIZE);
    double* labels_flat = flattenMatrix(labels, numImages, OUTPUT_SIZE);
    
    int correct = 0;
    
    for (int batch_start = 0; batch_start < numImages; batch_start += BATCH_SIZE) {
        int actual_batch_size = (batch_start + BATCH_SIZE <= numImages) ? BATCH_SIZE : (numImages - batch_start);
        
        double* batch_hidden = (double*)malloc(actual_batch_size * HIDDEN_SIZE * sizeof(double));
        double* batch_output = (double*)malloc(actual_batch_size * OUTPUT_SIZE * sizeof(double));
        
        forwardBatch(net, images_flat + batch_start * INPUT_SIZE, 
                     batch_hidden, batch_output, actual_batch_size);
        
        for (int b = 0; b < actual_batch_size; b++) {
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (batch_output[b * OUTPUT_SIZE + j] > batch_output[b * OUTPUT_SIZE + pred]) pred = j;
                if (labels_flat[(batch_start + b) * OUTPUT_SIZE + j] > labels_flat[(batch_start + b) * OUTPUT_SIZE + actual]) actual = j;
            }
            if (pred == actual) correct++;
        }
        
        free(batch_hidden);
        free(batch_output);
    }
    
    printf("Test Accuracy: %.2f%%\n", (correct / (double)numImages) * 100);
    
    free(images_flat);
    free(labels_flat);
}

void freeNetwork(NeuralNetwork* net) {
    freeMatrix(net->W1_host, HIDDEN_SIZE);
    freeMatrix(net->W2_host, OUTPUT_SIZE);
    free(net->b1_host);
    free(net->b2_host);
    
    cudaFree(net->W1_device);
    cudaFree(net->W2_device);
    cudaFree(net->b1_device);
    cudaFree(net->b2_device);
    
    free(net);
}

int main() {
    printf("MNIST Neural Network with CUDA\n\n");
    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "Error: No CUDA-capable device found\n");
        return 1;
    }
    
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Using GPU: %s\n", deviceProp.name);
    
    double** train_images = loadMNISTImages("data/train-images.idx3-ubyte", 60000);
    double** train_labels = loadMNISTLabels("data/train-labels.idx1-ubyte", 60000);
    double** test_images = loadMNISTImages("data/t10k-images.idx3-ubyte", 10000);
    double** test_labels = loadMNISTLabels("data/t10k-labels.idx1-ubyte", 10000);
    
    NeuralNetwork* net = createNetwork();
    train(net, train_images, train_labels, 60000);
    evaluate(net, test_images, test_labels, 10000);
    
    freeNetwork(net);
    freeMatrix(train_images, 60000);
    freeMatrix(train_labels, 60000);
    freeMatrix(test_images, 10000);
    freeMatrix(test_labels, 10000);

    cudaDeviceReset();
    
    return 0;
}