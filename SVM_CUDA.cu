// %%writefile SVM_CUDA.cu

// SVM implemented from scratch in CUDA

// Reference (Python) https://medium.com/@gallettilance/support-vector-machines-16241417ee6d
// Tyler Hilbert -- August 2nd, 2024


#ifndef __SVM_CUDA__
#define __SVM_CUDA__

#include <stdio.h>
#include <cuda_runtime.h>


#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Kernel that returns SVM update weights and update biases through d_w0_updates, d_w1_updates and d_b_updates
static __global__ void svm_update(
    const float* d_x0,
    const float* d_x1,
    const int* d_y,
    const float* d_w,
    const float* d_b,
    float* d_w0_updates,
    float* d_w1_updates,
    float* d_b_updates,
    const int n,
    const float learning_rate
) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float y = d_y[i];
        float prediction = d_w[0] * d_x0[i] + d_w[1] * d_x1[i] + *d_b;

        if (prediction * y <= 0 ) { 
            d_w0_updates[i] = d_x0[i] * y * learning_rate;
            d_w1_updates[i] = d_x1[i] * y * learning_rate;
            d_b_updates[i] = y * learning_rate;
        }
    }
}

// Kernel function to compute hinge loss
static __global__ void compute_loss(
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

class SVM_CUDA {
    public:

        SVM_CUDA(
            float *x0,
            float *x1,
            int *y,
            int train_size,
            int test_size,
            float learning_rate
        ) {

            this->train_size = train_size;
            this->test_size = test_size;
            this->learning_rate = learning_rate;

            // Initialize weights and bias
            h_w[0] = static_cast<float>(rand()) / RAND_MAX * 0.01;
            h_w[1] = static_cast<float>(rand()) / RAND_MAX * 0.01;
            h_b = 0.0f;

            // Block and grid size
            threads_per_block = 256;
            blocks_per_grid_train = (train_size + threads_per_block- 1) / threads_per_block;
            blocks_per_grid_test = (test_size + threads_per_block- 1) / threads_per_block;

            // CPU pointers
            h_x0 = x0;
            h_x1 = x1;
            h_y = y;

            // GPU malloc
            CUDA_CHECK( cudaMalloc(&d_x0_test, test_size * sizeof(float)));
            CUDA_CHECK( cudaMalloc(&d_x1_test, test_size * sizeof(float)));
            CUDA_CHECK( cudaMalloc(&d_x0_train, train_size * sizeof(float)));
            CUDA_CHECK( cudaMalloc(&d_x1_train, train_size * sizeof(float)));
            CUDA_CHECK( cudaMalloc(&d_y_test, test_size * sizeof(int)));
            CUDA_CHECK( cudaMalloc(&d_y_train, train_size * sizeof(int)));

            CUDA_CHECK( cudaMalloc(&d_w, 2 * sizeof(float)));
            CUDA_CHECK( cudaMalloc(&d_w0_updates, train_size * sizeof(float)));
            CUDA_CHECK( cudaMalloc(&d_w1_updates, train_size * sizeof(float)));
            CUDA_CHECK( cudaMalloc(&d_b, sizeof(float)));
            CUDA_CHECK( cudaMalloc(&d_b_updates, train_size * sizeof(float)));

            CUDA_CHECK( cudaMalloc(&d_test_loss, sizeof(float)));

            // Copy data from host to device
            CUDA_CHECK( cudaMemcpy(d_x0_train, h_x0, train_size * sizeof(float), cudaMemcpyHostToDevice) );
            CUDA_CHECK( cudaMemcpy(d_x0_test, (h_x0+train_size), test_size * sizeof(float), cudaMemcpyHostToDevice) );
            CUDA_CHECK( cudaMemcpy(d_x1_train, h_x1, train_size * sizeof(float), cudaMemcpyHostToDevice) );
            CUDA_CHECK( cudaMemcpy(d_x1_test, (h_x1+train_size), test_size * sizeof(float), cudaMemcpyHostToDevice) );
            CUDA_CHECK( cudaMemcpy(d_y_train, h_y, train_size * sizeof(int), cudaMemcpyHostToDevice) );
            CUDA_CHECK( cudaMemcpy(d_y_test, (h_y+train_size), test_size * sizeof(int), cudaMemcpyHostToDevice) );
            CUDA_CHECK( cudaMemcpy(d_w, h_w, 2 * sizeof(float), cudaMemcpyHostToDevice) );
            CUDA_CHECK( cudaMemcpy(d_b, &h_b, sizeof(float), cudaMemcpyHostToDevice) );
        }

        ~SVM_CUDA() {
            CUDA_CHECK( cudaFree(d_x0_test) );
            CUDA_CHECK( cudaFree(d_x1_test) );
            CUDA_CHECK( cudaFree(d_x0_train) );
            CUDA_CHECK( cudaFree(d_x1_train) );
            CUDA_CHECK( cudaFree(d_y_test) );
            CUDA_CHECK( cudaFree(d_y_train) );
            CUDA_CHECK( cudaFree(d_w) );
            CUDA_CHECK( cudaFree(d_w0_updates) );
            CUDA_CHECK( cudaFree(d_w1_updates) );
            CUDA_CHECK( cudaFree(d_b) );
            CUDA_CHECK( cudaFree(d_b_updates) );
            CUDA_CHECK( cudaFree(d_test_loss) );
        }

        void train_iteration() {
            float h_w0_updates[train_size];
            float h_w1_updates[train_size];
            float h_b_updates[train_size];

            // Launch kernel to get values to update weights and bias
            svm_update<<<blocks_per_grid_train, threads_per_block>>>(d_x0_train, d_x1_train, d_y_train, d_w, d_b, d_w0_updates, d_w1_updates, d_b_updates, train_size, learning_rate);
            CUDA_CHECK( cudaPeekAtLastError() );
            CUDA_CHECK( cudaDeviceSynchronize() );

            // update
            CUDA_CHECK( cudaMemcpy(&h_w0_updates, d_w0_updates, train_size * sizeof(float), cudaMemcpyDeviceToHost) );
            CUDA_CHECK( cudaMemcpy(&h_w1_updates, d_w1_updates, train_size * sizeof(float), cudaMemcpyDeviceToHost) );
            CUDA_CHECK( cudaMemcpy(&h_b_updates, d_b_updates, train_size * sizeof(float), cudaMemcpyDeviceToHost) );
            for (int i = 0; i < train_size; i++) {
                h_w[0] += h_w0_updates[i];
                h_w[1] += h_w1_updates[i];
                h_b += h_b_updates[i];
            }

            // Copy the weights and bias back to device
            CUDA_CHECK( cudaMemcpy(d_w, h_w, 2 * sizeof(float), cudaMemcpyHostToDevice) );
            CUDA_CHECK( cudaMemcpy(d_b, &h_b, sizeof(float), cudaMemcpyHostToDevice) );
        }


        void print() {
            // Calculate training loss
            // Reset loss value
            float h_test_loss = 0.0f;
            CUDA_CHECK( cudaMemcpy(d_test_loss, &h_test_loss, sizeof(float), cudaMemcpyHostToDevice) );

            // Launch kernel to calculate loss
            compute_loss<<<blocks_per_grid_test, threads_per_block>>>(d_x0_test, d_x1_test, d_y_test, d_w, d_b, d_test_loss, test_size);
            CUDA_CHECK( cudaPeekAtLastError() );
            CUDA_CHECK( cudaDeviceSynchronize() );

            CUDA_CHECK( cudaMemcpy(&h_test_loss, d_test_loss, sizeof(float), cudaMemcpyDeviceToHost) );

            // Print losses and model parameters
            printf("Test Loss: %f\n", h_test_loss / test_size);
            printf ("Weights:\n");
            printf ("float h_w[2] = {%f, %f};\n", h_w[0], h_w[1]);
            printf ("float h_b = %ff;", h_b);
            printf ("\n\n");
        }


    private:
        // Pointers to data
        float *h_x0, *h_x1;
        int *h_y;

        // Hyperparameters
        int train_size, test_size;
        float learning_rate;

        // Model
        float h_w[2];
        float h_b;

        // Block and grid size
        int threads_per_block;
        int blocks_per_grid_train;
        int blocks_per_grid_test;

        // GPU pointers
        float *d_x0_test, *d_x0_train, *d_x1_test, *d_x1_train;
        int *d_y_test, *d_y_train;
        float *d_w, *d_b;
        float *d_w0_updates, *d_w1_updates, *d_b_updates; 
        float *d_test_loss;
};

#endif // __SVM_CUDA__