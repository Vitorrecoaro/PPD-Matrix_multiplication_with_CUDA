#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <string.h>
#include <math.h>

void generateRandMatrix(double *matrix, int nElem);
void showMatrix(double *matrix, int nElem);
void multMatricesSeq(double *matrixA, double *matrixB, double *matrixC, int nElem);
__global__ void multMatricesGPU(double *matrixA, double *matrixB, double *matrixC, int nElem);
double averageError(double *matrixA, double *matrixB, int nElem);

int main(int argv, char *argc[]){
    int nElem = 5;
    int runSeq = 0;
    int runCUDA = 0;
    double *d_A, *d_B, *d_C, *h_A, *h_B, *h_C, *C_Cuda;
    dim3 block, grid;

    switch(argv){
        case(2):
            nElem = atoi(argc[1]);
            break;
        case(3):
            nElem = atoi(argc[1]);
            if (argc[2][0] == '-'){
                if (strchr(argc[2], 'C') != NULL){
                    runCUDA = 1;
                }
                if (strchr(argc[2], 'S') != NULL){
                    runSeq = 1;
                }
            }
            else{
                printf("You can run with this flags:\n");
                printf("    -S: To run sequential method.\n");
                printf("    -C: To run with cuda paralelization.\n");
            }
            break;
    }

    block.x = 32;
    block.y = 32;
    block.z = 1;

    grid.x = (nElem + block.x - 1)/block.x;
    grid.y = (nElem + block.y - 1)/block.y;
    grid.z = 1;

    cudaMalloc(&d_A, nElem * nElem * sizeof(double));
    cudaMalloc(&d_B, nElem * nElem * sizeof(double));
    cudaMalloc(&d_C, nElem * nElem * sizeof(double));
    h_A = (double*) malloc(nElem * nElem * sizeof(double));
    h_B = (double*) malloc(nElem * nElem * sizeof(double));
    h_C = (double*) malloc(nElem * nElem * sizeof(double));
    C_Cuda = (double*) malloc(nElem * nElem * sizeof(double));

    generateRandMatrix(h_A, nElem);
    generateRandMatrix(h_B, nElem);
    cudaMemcpy(d_A, h_A, nElem * nElem * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nElem * nElem * sizeof(double), cudaMemcpyHostToDevice);

    if (runCUDA){
        multMatricesGPU<<< grid, block >>>(d_A, d_B, d_C, nElem);
        cudaMemcpy(C_Cuda, d_C, nElem * nElem * sizeof(double), cudaMemcpyDeviceToHost);
    }

    if (runSeq){
        multMatricesSeq(h_A, h_B, h_C, nElem);
    }

    if (runCUDA && runSeq){
        printf("Average error: %f\n", averageError(C_Cuda, h_C, nElem));
        showMatrix(h_C, nElem);
        showMatrix(C_Cuda, nElem);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(C_Cuda);
    
}

// Calculate the multiplication of matrices on sequential form.
void multMatricesSeq(double *matrixA, double *matrixB, double *matrixC, int nElem){
    int i, j, k;

    for (i = 0; i < nElem; i++){
        for (k = 0; k < nElem; k++){
            matrixC[i * nElem + j] = 0.0;
            for (j = 0; j < nElem; j++){
                matrixC[i * nElem + j] += matrixA[i * nElem + k] * matrixB[k * nElem + j];
            }
        }
    }
}

// Calculate the multiplication of matrices with CUDA support.
__global__ void multMatricesGPU(double *matrixA, double *matrixB, double *matrixC, int nElem){
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i;
    double sum = 0;

    if(x < nElem && y < nElem){
        for (i = 0; i < nElem; i++){
            sum += matrixA[x * nElem + i] * matrixB[i * nElem + y];
        }
    }

    matrixC[x * nElem + y] = sum;
}

void showMatrix(double *matrix, int nElem){
    int i;

    for (i = 0; i < nElem * nElem; i++){
        printf("%f ", matrix[i]);
    }

    printf("\n");
}

// Generate Matrix with double values between [0...1000]
void generateRandMatrix(double *matrix, int nElem){
    int i;
    int tam = nElem * nElem;
    unsigned int seed = time(NULL);

    #pragma omp parallel for
    for (i = 0; i < tam; i++){
        matrix[i] = ((double) rand_r(&seed) / (double) RAND_MAX) * 1000;
    }
}

double averageError(double *matrixA, double *matrixB, int nElem){
    double errorAvg = 0;
    int i;
    
    #pragma omp parallel for reduction(+:errorAvg)
    for (i = 0; i < nElem * nElem; i++){
        if (matrixA[i] != matrixB[i]){
            errorAvg += abs(matrixA[i] - matrixB[i]);
        }
    }

    return (errorAvg/(nElem * nElem));
}