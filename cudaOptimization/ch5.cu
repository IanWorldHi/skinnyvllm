#define TILEWIDTH 16

__global__ void matricMulKernel(float* N, float* M, float* P, int width){
    __shared__ float Mds[TILEWIDTH][TILEWIDTH];
    __shared__ float Nds[TILEWIDTH][TILEWIDTH];
    int row = blockIdx.y * TILEWIDTH + threadIdx.y;
    int col = blockIdx.x * TILEWIDTH + threadIdx.x;
    
    float Pvalue = 0;
    for(int i = 0; i < width/TILEWIDTH; i++){
        
    }
}

int main(){
    
}



