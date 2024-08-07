// %%writefile PerformanceTester.cu

// Tests SVM class in CUDA

#include <chrono>
#include <vector>
#include <stdio.h>

#include "SVM_CUDA.cu"

#define N 100 // Number of datapoints in dataset
#define TRAIN_SIZE 75
#define TEST_SIZE (N - TRAIN_SIZE)

#define EPOCHS 100

#define LEARNING_RATE 0.05



int main() {
    std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();

    std::vector<std::chrono::time_point<std::chrono::system_clock>> p_marks(EPOCHS);
    std::vector<std::chrono::time_point<std::chrono::system_clock>> train_marks(EPOCHS);
    std::vector<std::chrono::time_point<std::chrono::system_clock>> mp_marks(EPOCHS);
    std::chrono::time_point<std::chrono::system_clock> cStart;
    std::chrono::time_point<std::chrono::system_clock> cEnd;

    // Linearly seperable dataset
    float x0[N] = { 1.23, 0.70, 2.55, 0.57, 0.59, 0.63, 0.56, 0.12, 1.32, 1.43, 0.64, 1.23, 0.74, 2.22, 3.08, 1.00, 1.03, 0.10, 0.58, 1.71, 1.16, 3.07, 1.89, 0.49, 3.08, 0.22, 1.13, 0.85, -0.01, 1.89, 0.77, 1.62, 2.21, 0.88, 0.60, 1.01, 2.31, 1.10, 0.76, 1.86, 1.69, 0.16, -0.08, 1.23, 1.29, -0.78, 0.51, 1.00, 1.43, 1.45, 0.73, 1.23, 1.05, 0.88, 0.79, 1.45, 1.22, 0.73, 0.09, 1.81, 0.71, 1.43, 0.80, 1.05, 0.38, 2.33, 1.81, 0.77, 1.84, 0.88, 0.59, 0.38, -0.21, 0.73, -0.40, 3.17, 0.38, 1.94, -0.23, 1.10, 0.65, 1.87, -0.86, 1.69, 1.86, 1.08, 0.34, 0.73, 1.44, 0.01, 0.25, 0.53, 1.41, 0.09, 1.57, 0.47, 1.23, 0.46, 0.68, 1.04 };
    float x1[N] = { -0.76, -1.38, 2.50, -1.35, -1.34, -1.17, -1.31, 0.20, -0.59, 1.39, -1.23, -0.73, 0.82, 2.10, 2.83, -1.03, -0.86, 0.22, 0.57, 1.57, -0.88, 2.85, -0.37, 0.60, 2.90, 0.29, 1.10, -1.18, -1.76, -0.37, -1.06, -0.57, 2.08, -1.18, -1.44, -0.96, 2.30, 1.10, -1.19, -0.41, 1.63, 0.14, -0.03, 1.20, -0.76, -0.64, -1.29, -1.05, -0.71, -0.74, 0.69, 1.20, -1.10, -1.15, -1.24, 1.39, -0.96, 0.69, 0.09, -0.27, 0.84, 1.35, 0.74, 0.97, -1.58, 2.26, 1.73, 0.75, -0.33, -1.14, 0.66, -1.56, -0.21, -1.25, -2.26, 2.97, -1.48, 1.90, -0.15, -0.93, 0.69, -0.37, -0.73, -0.36, 1.78, -0.96, -1.45, -1.14, 1.41, 0.12, 0.28, -1.52, -0.68, 0.20, 1.54, -1.44, 1.24, -1.49, 0.76, 1.11 };
    int y[N] = { -1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1 };

    cStart = std::chrono::system_clock::now();
    SVM_CUDA model(x0, x1, y, TRAIN_SIZE, TEST_SIZE, LEARNING_RATE);
    cEnd = std::chrono::system_clock::now();

    for (int i = 0; i < EPOCHS; i++) {
        printf ("EPOCH: %i\n", i);
        p_marks.at(i) = std::chrono::system_clock::now();
        model.train_iteration();
        train_marks.at(i) = std::chrono::system_clock::now();
        model.print();
        mp_marks.at(i) = std::chrono::system_clock::now();;
    }

    // Print performance
    printf ("Constructor:\t%ld\n", std::chrono::duration_cast<std::chrono::nanoseconds>(cEnd - cStart).count());
    for (int i = 0; i < EPOCHS; i++) {
        printf ("Train:\t%ld\n", std::chrono::duration_cast<std::chrono::nanoseconds>(train_marks.at(i) - p_marks.at(i)).count());
        printf ("Test:\t%ld\n\n", std::chrono::duration_cast<std::chrono::nanoseconds>(mp_marks.at(i) - train_marks.at(i)).count());
    }
}