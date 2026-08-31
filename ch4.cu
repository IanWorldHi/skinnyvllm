#include <stdio.h>
#include <iostream>
using namespace std;

int main(){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    cout << deviceCount << endl;
}



