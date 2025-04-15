// naive_cuda_nn.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01f
#define EPOCHS 1
#define NUM_CLASSES 10
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
__device__ float relu(float x) {
    return (x > 0) ? x : 0;
}

__device__ float relu_derivative(float x) {
    return (x > 0) ? 1.0f : 0.0f;
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
__global__ void softmaxKernel(double* A, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        double max_val = A[row * cols];
        for (int i = 1; i < cols; i++) {
            if (A[row * cols + i] > max_val) {
                max_val = A[row * cols + i];
            }
        }
        
        double sum = 0.0;
        for (int i = 0; i < cols; i++) {
            double val = A[row * cols + i] - max_val;
            if (val > -708.0) { // log(DBL_MIN) is around -708
                A[row * cols + i] = exp(val);
            } else {
                A[row * cols + i] = 0.0;
            }
            sum += A[row * cols + i];
        }
        
        const double eps = 1e-15;
        for (int i = 0; i < cols; i++) {
            A[row * cols + i] /= (sum + eps);
        }
    }
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

// Convert flattened 1D array to 2D matrix
void unflattenMatrix(double* flat, double** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = flat[i * cols + j];
        }
    }
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
// Main function
int main() {
    printf("MNIST Neural Network with CUDA\n\n");
    
    // Check for CUDA device
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