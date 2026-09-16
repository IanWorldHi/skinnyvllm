#include <cstdio>
#include <cstdlib>
#include <cmath>

#define TILEWIDTH 16

__global__ void matricMulKernel(float* N, float* M, float* P, int width){
    __shared__ float Mds[TILEWIDTH][TILEWIDTH];
    __shared__ float Nds[TILEWIDTH][TILEWIDTH];
    int row = blockIdx.y * TILEWIDTH + threadIdx.y;
    int col = blockIdx.x * TILEWIDTH + threadIdx.x;
    
    float Pvalue = 0;
    for(int i = 0; i < width/TILEWIDTH; i++){
        Mds[threadIdx.y][threadIdx.x] = M[row*width + (i*TILEWIDTH + threadIdx.x)];
        Nds[threadIdx.y][threadIdx.x] = N[(i*TILEWIDTH + threadIdx.y)*width + col];
        __syncthreads();
        for(int j = 0; j < TILEWIDTH; j++){
            Pvalue += Mds[threadIdx.y][j] * Nds[j][threadIdx.x];
        }
        __syncthreads();
    }
    P[row*width + col] = Pvalue;
}

int main(){
    const int width = 512; // must be a multiple of TILEWIDTH
    const size_t bytes = (size_t)width * width * sizeof(float);

    float *h_M = (float*)malloc(bytes);
    float *h_N = (float*)malloc(bytes);
    float *h_P = (float*)malloc(bytes);

    for (int i = 0; i < width * width; i++) {
        h_M[i] = 1.0f;
        h_N[i] = 2.0f;
    }

    float *d_M, *d_N, *d_P;
    cudaMalloc(&d_M, bytes);
    cudaMalloc(&d_N, bytes);
    cudaMalloc(&d_P, bytes);

    cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice);

    dim3 blockDim(TILEWIDTH, TILEWIDTH);
    dim3 gridDim(width / TILEWIDTH, width / TILEWIDTH);
    matricMulKernel<<<gridDim, blockDim>>>(d_N, d_M, d_P, width);

    cudaMemcpy(h_P, d_P, bytes, cudaMemcpyDeviceToHost);

    // expected value for every element: width * 1.0 * 2.0
    float expected = width * 2.0f;
    bool ok = true;
    for (int i = 0; i < width * width; i++) {
        if (fabs(h_P[i] - expected) > 1e-3) {
            printf("Mismatch at %d: got %f, expected %f\n", i, h_P[i], expected);
            ok = false;
            break;
        }
    }
    printf(ok ? "PASS\n" : "FAIL\n");

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    free(h_M);
    free(h_N);
    free(h_P);

    return 0;
}



