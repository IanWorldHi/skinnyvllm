#include <stdio.h>
#include <iostream>
#include <math.h>


__global__ void add(int n, float *sum, float *x, float *y) {
    int index = threadIdx.x;
    int stride = blockDim.x;
    for(int i = index; i<n; i+=stride){
        sum[i] = x[i] + y[i];
    }
    //each thread starts at a different index, then skips by the number of threads, effectively splitting the work
    //ie start at i = 6, skip to i = 6 + stride

    /* for(int i = 0; i<n; i++){
        sum[i] = x[i] + y[i];
    } */
}

int main() {
    printf("Hello, World!\n");

    int N = 1<<20; //2 to the 20th bitshift
    float *x, *y, *sum;
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));
    cudaMallocManaged(&sum, N*sizeof(float));

    for(int i = 0; i<N; i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    add<<<1, 256>>>(N, sum, x, y);

    cudaDeviceSynchronize();

    float maxError = 0.0f;
    for(int i = 0; i<N; i++){
        maxError = fmax(maxError, fabs(sum[i]-3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    cudaFree(x);
    cudaFree(y);
    cudaFree(sum);

    return 0;
}







