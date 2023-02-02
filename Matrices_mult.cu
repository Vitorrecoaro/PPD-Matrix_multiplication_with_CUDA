#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

// Generate Matrix with float values between [0...1000]
void generateRandMatrix(float *matrix, int nElem){
    int i;
    int tam = nElem * nElem;
    unsigned int seed = time(NULL);

    #pragma omp parallel for
    for (i = 0; i < tam; i++){
        matrix[i] = ((float) rand_r(&seed) / (float) RAND_MAX) * 1000;
    }
}

// Calculate a multiplication of matrices on sequential form.
void multMatricesSeq(float *matrixA, float *matrixB, float *matrixC, int nElem){
    int i, j, k;

    for (i = 0; i < nElem; i++){
        for (j = 0; j < nElem; j++){
            for (k = 0; k < nElem; k++){
                matrixC[i * nElem + j] += matrixA[i * nElem + k] * matrixB[k * nElem + j];
            }
        }
    }
}

void showMatrix(float *matrix, int nElem){
    int i;

    for (i = 0; i < nElem * nElem; i++){
        printf("%f ", matrix[i]);
    }

    printf("\n");
}

int main(){
    int nElem = 2;
    float *A, *B, *C;

    cudaMallocManaged(&A, nElem * nElem * sizeof(float));
    cudaMallocManaged(&B, nElem * nElem * sizeof(float));
    cudaMallocManaged(&C, nElem * nElem * sizeof(float));

    generateRandMatrix(A, nElem);
    generateRandMatrix(B, nElem);

    multMatricesSeq(A, B, C, nElem);
    showMatrix(A, nElem);
    showMatrix(B, nElem);
    showMatrix(C, nElem);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    
}