// SVM implemented from scratch in CUDA
// Reference (Python) https://medium.com/@gallettilance/support-vector-machines-16241417ee6d
// Tyler Hilbert -- August 2nd, 2024

#include <stdio.h>
#include <cuda_runtime.h>

#define N 100 // Number of datapoints in dataset
#define TRAIN_SIZE 75
#define TEST_SIZE (N - TRAIN_SIZE)
#define D 2 // Number of dimensions

#define EPOCHS 500
#define PRINT_FREQ 25 // How many epochs to print loss and weights after

#define LEARNING_RATE 0.0001
#define EXPANDING_RATE .9999
#define RETRACTING_RATE 1.0001


// Kernel that returns SVM update weights and update biases through d_w0_updates, d_w1_updates and d_b_updates
__global__ void svm_update(
    const float* d_x0,
    const float* d_x1,
    const int* d_y,
    const float* d_w,
    const float* d_b,
    float* d_w0_updates,
    float* d_w1_updates,
    float* d_b_updates
) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < TRAIN_SIZE) {
        float y = d_y[i];
        float prediction = d_w[0] * d_x0[i] + d_w[1] * d_x1[i] + *d_b;

        if ( (prediction > 0 and y > 0) || (prediction < 0 and y < 0) ) { // Correct prediction
            if (prediction < 1 && prediction > -1) { // Street too wide
                float correction = y * LEARNING_RATE * RETRACTING_RATE;
                d_w0_updates[i] = d_x0[i] * correction;
                d_w1_updates[i] = d_x1[i] * correction;
                d_b_updates[i] = correction;
            } else { // Street too narrow
                d_w0_updates[i] = d_w[0] * (1-EXPANDING_RATE);
                d_w1_updates[i] = d_w[1] * (1-EXPANDING_RATE);
                d_b_updates[i] = *d_b * (1-EXPANDING_RATE);
            }
        } else { // Misclassified
            float correction = y * LEARNING_RATE * EXPANDING_RATE;
            d_w0_updates[i] = d_x0[i] * correction;
            d_w1_updates[i] = d_x1[i] * correction;
            d_b_updates[i] = correction;
        }
    }
}

// Kernel function to compute hinge loss
__global__ void compute_loss(
    const float* d_x0,
    const float* d_x1,
    const int* d_y,
    const float* d_w,
    const float* d_b,
    float* d_loss,
    const int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {

        // Compute the decision value
        float decision = d_w[0] * d_x0[i] + d_w[1] * d_x1[i] + *d_b;

        // Compute hinge loss
        float loss = fmaxf(0.0f, 1.0f - d_y[i] * decision);
        atomicAdd(d_loss, loss);
    }
}

int main() {
    // Linearly seperable dataset
    float h_x1[N] = { 1.23, 0.70, 2.55, 0.57, 0.59, 0.63, 0.56, 0.12, 1.32, 1.43, 0.64, 1.23, 0.74, 2.22, 3.08, 1.00, 1.03, 0.10, 0.58, 1.71, 1.16, 3.07, 1.89, 0.49, 3.08, 0.22, 1.13, 0.85, -0.01, 1.89, 0.77, 1.62, 2.21, 0.88, 0.60, 1.01, 2.31, 1.10, 0.76, 1.86, 1.69, 0.16, -0.08, 1.23, 1.29, -0.78, 0.51, 1.00, 1.43, 1.45, 0.73, 1.23, 1.05, 0.88, 0.79, 1.45, 1.22, 0.73, 0.09, 1.81, 0.71, 1.43, 0.80, 1.05, 0.38, 2.33, 1.81, 0.77, 1.84, 0.88, 0.59, 0.38, -0.21, 0.73, -0.40, 3.17, 0.38, 1.94, -0.23, 1.10, 0.65, 1.87, -0.86, 1.69, 1.86, 1.08, 0.34, 0.73, 1.44, 0.01, 0.25, 0.53, 1.41, 0.09, 1.57, 0.47, 1.23, 0.46, 0.68, 1.04 };
    float h_x2[N] = { -0.76, -1.38, 2.50, -1.35, -1.34, -1.17, -1.31, 0.20, -0.59, 1.39, -1.23, -0.73, 0.82, 2.10, 2.83, -1.03, -0.86, 0.22, 0.57, 1.57, -0.88, 2.85, -0.37, 0.60, 2.90, 0.29, 1.10, -1.18, -1.76, -0.37, -1.06, -0.57, 2.08, -1.18, -1.44, -0.96, 2.30, 1.10, -1.19, -0.41, 1.63, 0.14, -0.03, 1.20, -0.76, -0.64, -1.29, -1.05, -0.71, -0.74, 0.69, 1.20, -1.10, -1.15, -1.24, 1.39, -0.96, 0.69, 0.09, -0.27, 0.84, 1.35, 0.74, 0.97, -1.58, 2.26, 1.73, 0.75, -0.33, -1.14, 0.66, -1.56, -0.21, -1.25, -2.26, 2.97, -1.48, 1.90, -0.15, -0.93, 0.69, -0.37, -0.73, -0.36, 1.78, -0.96, -1.45, -1.14, 1.41, 0.12, 0.28, -1.52, -0.68, 0.20, 1.54, -1.44, 1.24, -1.49, 0.76, 1.11 };
    int h_y[N] = { -1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1 };

    // Initialize weights and bias
    float h_w[D] = {0.0f, 0.0f};
    float h_b = 0.0f;

    // Block and grid size
    int threads_per_block= 256;
    int blocks_per_grid_train = (TRAIN_SIZE + threads_per_block- 1) / threads_per_block;
    int blocks_per_grid_test = (TEST_SIZE + threads_per_block- 1) / threads_per_block;

    // GPU Memory
    float *d_x1, *d_x2, *d_w, *d_w0_updates, *d_w1_updates, *d_b, *d_b_updates, *d_train_loss, *d_test_loss;
    int *d_y;

    cudaError_t err;
    cudaMalloc(&d_x1, N * sizeof(float));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("1CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaMalloc(&d_x2, N * sizeof(float));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("2CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaMalloc(&d_y, N * sizeof(int));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("3CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaMalloc(&d_w, D * sizeof(float));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("4CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaMalloc(&d_w0_updates, TRAIN_SIZE * sizeof(float));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("4.aCUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaMalloc(&d_w1_updates, TRAIN_SIZE * sizeof(float));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("4.bCUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaMalloc(&d_b, sizeof(float));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("5CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaMalloc(&d_b_updates, TRAIN_SIZE * sizeof(float));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("5.aCUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaMalloc(&d_train_loss, sizeof(float));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("6CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaMalloc(&d_test_loss, sizeof(float));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("7CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpy(d_x1, h_x1, N * sizeof(float), cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("8CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpy(d_x2, h_x2, N * sizeof(float), cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("9CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpy(d_y, h_y, N * sizeof(int), cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("10CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpy(d_w, h_w, D * sizeof(float), cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("11CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpy(d_b, &h_b, sizeof(float), cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("12CUDA error: %s\n", cudaGetErrorString(err));
    }


    // Training loop
    float h_w0_updates[TRAIN_SIZE];
    float h_w1_updates[TRAIN_SIZE];
    float h_b_updates[TRAIN_SIZE];
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        // Launch kernel to get values to update weights and bias
        svm_update<<<blocks_per_grid_train, threads_per_block>>>(d_x1, d_x2, d_y, d_w, d_b, d_w0_updates, d_w1_updates, d_b_updates);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
          printf("13CUDA error: %s\n", cudaGetErrorString(err));
        }
        // update
        cudaMemcpy(&h_w0_updates, d_w0_updates, TRAIN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("13.aCUDA error: %s\n", cudaGetErrorString(err));
        }
        for (int i = 0; i < TRAIN_SIZE; i++) {
            h_w[0] += h_w0_updates[i];
            //printf ("h_w0_updates[i] %f\n", h_w0_updates[i]);
        }
        cudaMemcpy(&h_w1_updates, d_w1_updates, TRAIN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("13.bCUDA error: %s\n", cudaGetErrorString(err));
        }
        for (int i = 0; i < TRAIN_SIZE; i++) {
            h_w[1] += h_w1_updates[i];
        }
        cudaMemcpy(&h_b_updates, d_b_updates, TRAIN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("13.cCUDA error: %s\n", cudaGetErrorString(err));
        }
        for (int i = 0; i < TRAIN_SIZE; i++) {
            h_b += h_b_updates[i];
        }

        // Copy the weights and bias back to device
        cudaMemcpy(d_w, h_w, D * sizeof(float), cudaMemcpyHostToDevice);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
          printf("19CUDA error: %s\n", cudaGetErrorString(err));
        }
        cudaMemcpy(d_b, &h_b, sizeof(float), cudaMemcpyHostToDevice);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
          printf("20CUDA error: %s\n", cudaGetErrorString(err));
        }

        if (epoch % PRINT_FREQ == 0 || (epoch+1) == EPOCHS) {
            // Calculate training loss
            // Reset loss values
            float h_train_loss = 0.0f;
            float h_test_loss = 0.0f;
            cudaMemcpy(d_train_loss, &h_train_loss, sizeof(float), cudaMemcpyHostToDevice);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
              printf("14CUDA error: %s\n", cudaGetErrorString(err));
            }
            cudaMemcpy(d_test_loss, &h_test_loss, sizeof(float), cudaMemcpyHostToDevice);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
              printf("15CUDA error: %s\n", cudaGetErrorString(err));
            }

            // Launch kernel to calculate train loss
            compute_loss<<<blocks_per_grid_train, threads_per_block>>>(d_x1, d_x2, d_y, d_w, d_b, d_train_loss, TRAIN_SIZE);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
              printf("16CUDA error: %s\n", cudaGetErrorString(err));
            }
            cudaMemcpy(&h_train_loss, d_train_loss, sizeof(float), cudaMemcpyDeviceToHost);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
              printf("17CUDA error: %s\n", cudaGetErrorString(err));
            }

            // Launch kernel to calculate test loss
            compute_loss<<<blocks_per_grid_test, threads_per_block>>>(d_x1 + TRAIN_SIZE, d_x2 + TRAIN_SIZE, d_y + TRAIN_SIZE, d_w, d_b, d_test_loss, TEST_SIZE);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
              printf("18CUDA error: %s\n", cudaGetErrorString(err));
            }
            cudaMemcpy(&h_test_loss, d_test_loss, sizeof(float), cudaMemcpyDeviceToHost);

            // Print losses and model parameters
            printf("Epoch: %i\nTrain Loss: %f\nTest Loss: %f\n", epoch, (h_train_loss / TRAIN_SIZE), (h_test_loss / TEST_SIZE));
            printf ("Weights:\n");
            printf ("float h_w[D] = {%f, %f};\n", h_w[0], h_w[1]);
            printf ("float h_b = %ff;", h_b);
            printf ("\n\n");
        }
    }

    // Free device memory
    cudaFree(d_x1);
    cudaFree(d_x2);
    cudaFree(d_y);
    cudaFree(d_w);
    cudaFree(d_b);
    cudaFree(d_train_loss);
    cudaFree(d_test_loss);
    cudaFree(d_w0_updates);
    cudaFree(d_w1_updates);
    cudaFree(d_b_updates);

    return 0;
}