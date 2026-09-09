#include <stdio.h>
#include <iostream>
using namespace std;

int main(){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    cout << deviceCount << endl;
    cout << deviceProp.multiProcessorCount << " " << deviceProp.maxThreadsPerBlock << endl;
    //cout << deviceProp.clockRate << " " << deviceProp.maxThreadsDim[0] << endl;
    //clock rate motived to cudaDeviceAttr instead of cudaDeviceProp?
}








