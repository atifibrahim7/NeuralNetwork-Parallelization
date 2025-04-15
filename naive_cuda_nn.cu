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

__device__ float relu(float x) {
    return (x > 0) ? x : 0;
}

__device__ float relu_derivative(float x) {
    return (x > 0) ? 1.0f : 0.0f;
}

__device__ void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; ++i)
        if (x[i] > max_val) max_val = x[i];

    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        x[i] = expf(x[i] - max_val);  
        sum += x[i];
    }
    for (int i = 0; i < size; ++i) {
        x[i] /= sum;
    }
}

__global__ void trainKernel(float* d_images, float* d_labels,
                            float* W1, float* b1, float* W2, float* b2,
                            int numSamples) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSamples) return;

    float input[INPUT_SIZE];
    float hidden[HIDDEN_SIZE];
    float output[OUTPUT_SIZE];
    float d_hidden[HIDDEN_SIZE];
    float d_output[OUTPUT_SIZE];

    for (int i = 0; i < INPUT_SIZE; ++i)
        input[i] = d_images[idx * INPUT_SIZE + i];

    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        float sum = b1[i];
        for (int j = 0; j < INPUT_SIZE; ++j) {
            sum += W1[i * INPUT_SIZE + j] * input[j];
        }
        hidden[i] = relu(sum);
    }

    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        float sum = b2[i];
        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            sum += W2[i * HIDDEN_SIZE + j] * hidden[j];
        }
        output[i] = sum;
    }

    softmax(output, OUTPUT_SIZE);

    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        d_output[i] = output[i] - d_labels[idx * OUTPUT_SIZE + i];
    }

    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        d_hidden[i] = 0;
        for (int j = 0; j < OUTPUT_SIZE; ++j) {
            d_hidden[i] += W2[j * HIDDEN_SIZE + i] * d_output[j];
        }
        d_hidden[i] *= relu_derivative(hidden[i]);
    }

    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            W2[i * HIDDEN_SIZE + j] -= LEARNING_RATE * d_output[i] * hidden[j];
        }
        b2[i] -= LEARNING_RATE * d_output[i];
    }

    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        for (int j = 0; j < INPUT_SIZE; ++j) {
            W1[i * INPUT_SIZE + j] -= LEARNING_RATE * d_hidden[i] * input[j];
        }
        b1[i] -= LEARNING_RATE * d_hidden[i];
    }
}

float* loadFileToArray(const char* filename, int numItems, int itemSize, int offset) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Cannot open %s\n", filename);
        exit(1);
    }
    fseek(file, offset, SEEK_SET);
    float* data = (float*)malloc(sizeof(float) * numItems * itemSize);
    for (int i = 0; i < numItems * itemSize; ++i) {
        unsigned char byte;
        fread(&byte, sizeof(unsigned char), 1, file);
        data[i] = byte / 255.0f;
    }
    fclose(file);
    return data;
}

float* loadLabelsToOneHot(const char* filename, int numItems) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Cannot open %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    float* labels = (float*)calloc(numItems * OUTPUT_SIZE, sizeof(float));
    for (int i = 0; i < numItems; ++i) {
        unsigned char label;
        fread(&label, sizeof(unsigned char), 1, file);
        labels[i * OUTPUT_SIZE + label] = 1.0f;
    }
    fclose(file);
    return labels;
}

int main() {
    int train_size = 10000;  // Keep small for demo

    printf("Loading MNIST data...\n");
    float* h_train_images = loadFileToArray("data/train-images.idx3-ubyte", train_size, INPUT_SIZE, 16);
    float* h_train_labels = loadLabelsToOneHot("data/train-labels.idx1-ubyte", train_size);

    // Allocate GPU memory
    float *d_images, *d_labels;
    cudaMalloc(&d_images, train_size * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_labels, train_size * OUTPUT_SIZE * sizeof(float));
    cudaMemcpy(d_images, h_train_images, train_size * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_train_labels, train_size * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate network
    float *W1, *b1, *W2, *b2;
    cudaMallocManaged(&W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMallocManaged(&b1, HIDDEN_SIZE * sizeof(float));
    cudaMallocManaged(&W2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    cudaMallocManaged(&b2, OUTPUT_SIZE * sizeof(float));

    srand(time(NULL));
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) W1[i] = ((float)rand() / RAND_MAX) * 0.01f;
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) W2[i] = ((float)rand() / RAND_MAX) * 0.01f;
    for (int i = 0; i < HIDDEN_SIZE; i++) b1[i] = 0.0f;
    for (int i = 0; i < OUTPUT_SIZE; i++) b2[i] = 0.0f;

    // Train
    printf("Training...\n");
    int threads = 256;
    int blocks = (train_size + threads - 1) / threads;
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        trainKernel<<<blocks, threads>>>(d_images, d_labels, W1, b1, W2, b2, train_size);
        cudaDeviceSynchronize();
        printf("Epoch %d complete\n", epoch + 1);
    }

    // Cleanup
    cudaFree(d_images);
    cudaFree(d_labels);
    cudaFree(W1);
    cudaFree(b1);
    cudaFree(W2);
    cudaFree(b2);
    free(h_train_images);
    free(h_train_labels);

    return 0;
}
