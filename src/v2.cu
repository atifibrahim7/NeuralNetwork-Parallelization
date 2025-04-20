#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
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
    
    double *d_W1, *d_W2;
    double *d_b1, *d_b2;
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

__global__ void relu_kernel(double* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = (x[idx] > 0) ? x[idx] : 0;
    }
}

__global__ void softmax_kernel(double* x, int size) {
    __shared__ double max_val;
    if (threadIdx.x == 0) {
        max_val = -INFINITY;
        for (int i = 0; i < size; i++) {
            if (x[i] > max_val) max_val = x[i];
        }
    }
    __syncthreads();
    
    __shared__ double sum;
    if (threadIdx.x == 0) {
        sum = 0.0;
        for (int i = 0; i < size; i++) {
            x[i] = exp(x[i] - max_val);
            sum += x[i];
        }
    }
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] /= sum;
    }
}

__global__ void forward_hidden_kernel(double* W1, double* input, double* b1, double* hidden, int hidden_size, int input_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < hidden_size) {
        hidden[i] = b1[i];
        for (int j = 0; j < input_size; j++) {
            hidden[i] += W1[i * input_size + j] * input[j];
        }
    }
}

__global__ void forward_output_kernel(double* W2, double* hidden, double* b2, double* output, int output_size, int hidden_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < output_size) {
        output[i] = b2[i];
        for (int j = 0; j < hidden_size; j++) {
            output[i] += W2[i * hidden_size + j] * hidden[j];
        }
    }
}

__global__ void compute_output_gradient_kernel(double* output, double* target, double* d_output, int output_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < output_size) {
        d_output[i] = output[i] - target[i];
    }
}

__global__ void compute_hidden_gradient_kernel(double* W2, double* d_output, double* hidden, double* d_hidden, 
                                              int hidden_size, int output_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < hidden_size) {
        d_hidden[i] = 0;
        for (int j = 0; j < output_size; j++) {
            d_hidden[i] += W2[j * hidden_size + i] * d_output[j];
        }
        d_hidden[i] *= (hidden[i] > 0);
    }
}

__global__ void update_output_weights_kernel(double* W2, double* d_output, double* hidden, 
                                            int output_size, int hidden_size, double learning_rate) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < output_size && j < hidden_size) {
        W2[i * hidden_size + j] -= learning_rate * d_output[i] * hidden[j];
    }
}

__global__ void update_hidden_weights_kernel(double* W1, double* d_hidden, double* input, 
                                            int hidden_size, int input_size, double learning_rate) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < hidden_size && j < input_size) {
        W1[i * input_size + j] -= learning_rate * d_hidden[i] * input[j];
    }
}

__global__ void update_output_bias_kernel(double* b2, double* d_output, int output_size, double learning_rate) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < output_size) {
        b2[i] -= learning_rate * d_output[i];
    }
}

__global__ void update_hidden_bias_kernel(double* b1, double* d_hidden, int hidden_size, double learning_rate) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < hidden_size) {
        b1[i] -= learning_rate * d_hidden[i];
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

    CHECK_CUDA_ERROR(cudaMemcpy(net->d_W1, flat_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_W2, flat_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_b1, net->b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_b2, net->b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    
    free(flat_W1);
    free(flat_W2);
    
    printf("GPU memory allocation and transfer time: %.3fs\n", get_time(mem_start));

    return net;
}

void forward(NeuralNetwork* net, double* input, double* hidden, double* output, double* debugTimes) {
    clock_t transfer_start = clock();
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_input, input, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    debugTimes[0] += get_time(transfer_start); // Input transfer time

    clock_t compute_start = clock();
    
    int blockSize = 256;
    int numBlocks = (HIDDEN_SIZE + blockSize - 1) / blockSize;
    forward_hidden_kernel<<<numBlocks, blockSize>>>(net->d_W1, net->d_input, net->d_b1, net->d_hidden, 
                                                   HIDDEN_SIZE, INPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    relu_kernel<<<numBlocks, blockSize>>>(net->d_hidden, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    numBlocks = (OUTPUT_SIZE + blockSize - 1) / blockSize;
    forward_output_kernel<<<numBlocks, blockSize>>>(net->d_W2, net->d_hidden, net->d_b2, net->d_output, 
                                                   OUTPUT_SIZE, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    softmax_kernel<<<1, 32>>>(net->d_output, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    debugTimes[1] += get_time(compute_start); // Forward computation time
    
    transfer_start = clock();
    CHECK_CUDA_ERROR(cudaMemcpy(hidden, net->d_hidden, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(output, net->d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    debugTimes[2] += get_time(transfer_start); // Output transfer time
}

void backward(NeuralNetwork* net, double* input, double* hidden, double* output, double* target, double* debugTimes) {
    clock_t transfer_start = clock();
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_input, input, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_hidden, hidden, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_output, output, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_target, target, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    debugTimes[3] += get_time(transfer_start); // Backward transfer time
    
    clock_t compute_start = clock();
    
    int blockSize = 256;
    
    int numBlocks = (OUTPUT_SIZE + blockSize - 1) / blockSize;
    compute_output_gradient_kernel<<<numBlocks, blockSize>>>(net->d_output, net->d_target, 
                                                            net->d_d_output, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    numBlocks = (HIDDEN_SIZE + blockSize - 1) / blockSize;
    compute_hidden_gradient_kernel<<<numBlocks, blockSize>>>(net->d_W2, net->d_d_output, net->d_hidden, 
                                                           net->d_d_hidden, HIDDEN_SIZE, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Update weights using 2D grid
    dim3 blockDim(16, 16);
    dim3 gridDim_output((OUTPUT_SIZE + blockDim.x - 1) / blockDim.x, 
                        (HIDDEN_SIZE + blockDim.y - 1) / blockDim.y);
    update_output_weights_kernel<<<gridDim_output, blockDim>>>(net->d_W2, net->d_d_output, net->d_hidden, 
                                                            OUTPUT_SIZE, HIDDEN_SIZE, LEARNING_RATE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    dim3 gridDim_hidden((HIDDEN_SIZE + blockDim.x - 1) / blockDim.x, 
                        (INPUT_SIZE + blockDim.y - 1) / blockDim.y);
    update_hidden_weights_kernel<<<gridDim_hidden, blockDim>>>(net->d_W1, net->d_d_hidden, net->d_input, 
                                                            HIDDEN_SIZE, INPUT_SIZE, LEARNING_RATE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    numBlocks = (OUTPUT_SIZE + blockSize - 1) / blockSize;
    update_output_bias_kernel<<<numBlocks, blockSize>>>(net->d_b2, net->d_d_output, 
                                                      OUTPUT_SIZE, LEARNING_RATE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    numBlocks = (HIDDEN_SIZE + blockSize - 1) / blockSize;
    update_hidden_bias_kernel<<<numBlocks, blockSize>>>(net->d_b1, net->d_d_hidden, 
                                                      HIDDEN_SIZE, LEARNING_RATE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    debugTimes[4] += get_time(compute_start); // Backward computation time
    
    transfer_start = clock();
    
    double* flat_W1 = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    double* flat_W2 = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    
    CHECK_CUDA_ERROR(cudaMemcpy(flat_W1, net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(flat_W2, net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(net->b1, net->d_b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(net->b2, net->d_b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    
    unflatten_matrix(flat_W1, net->W1, HIDDEN_SIZE, INPUT_SIZE);
    unflatten_matrix(flat_W2, net->W2, OUTPUT_SIZE, HIDDEN_SIZE);
    
    free(flat_W1);
    free(flat_W2);
    
    debugTimes[5] += get_time(transfer_start); // Weights transfer time
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
    printf("Starting training on GPU...\n");
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int i = 0; i < 6; i++) {
            debugTimes[i] = 0;
        }
        
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;

        for (int i = 0; i < numImages; i++) {
            double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
            forward(net, images[i], hidden, output, debugTimes);
            backward(net, images[i], hidden, output, labels[i], debugTimes);

            for (int k = 0; k < OUTPUT_SIZE; k++) loss -= labels[i][k] * log(output[k]);
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[pred]) pred = j;
                if (labels[i][j] > labels[i][actual]) actual = j;
            }
            if (pred == actual) correct++;
        }

        double epoch_time = get_time(epoch_start);
        double accuracy = (correct / (double)numImages) * 100;
        
        printf("\n====================================================\n");
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
    CHECK_CUDA_ERROR(cudaFree(net->d_W1));
    CHECK_CUDA_ERROR(cudaFree(net->d_W2));
    CHECK_CUDA_ERROR(cudaFree(net->d_b1));
    CHECK_CUDA_ERROR(cudaFree(net->d_b2));
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
    printf("MNIST Neural Network (CUDA Implementation)\n\n");
    
    cudaDeviceProp prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));
    printf("Using GPU: %s\n", prop.name);

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
    
    return 0;
}
