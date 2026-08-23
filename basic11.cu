#include <stdio.h>
#include <iostream>
#include <math.h>

//Inefficent but conceptual overview again of vector addition on gpu

#define ERROR_CUDA_EXAMPLE(...) {fprintf()}

__global__ add2(){

}


void add(int n, float *A_h, float *B_h, float *C_h) {
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    
    cudaError_t err = cudaMalloc((void **) &A_d, size);
    if(error!=cudaSuccess){
        printf("%s in %s on line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE); //vs exit(1) again?
    }

    cudaMalloc((void **) &A_d, size);
    cudaMalloc((void **) &B_d, size);
    cudaMalloc((void **) &C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    add2<<<ceil(n/256.0), 256>>>();

    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
    
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    
}







