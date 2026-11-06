#include <stdio.h>
#include <iostream>
#include <math.h>
#include <fstream>
#include <inttypes.h> //for uint64_t, gives PRIu64 macro for printing uint64_t

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <queue>
#define JSON_USE_IMPLICIT_CONVERSIONS 0 // this one is useful, but I don't remember why
#include "json.hpp"

using json = nlohmann::json;


constexpr size_t B_TO_MB = 1024ull * 1024;
constexpr size_t B_TO_GB = 1024ull * 1024 * 1024;
constexpr int NUM_LAYERS = 16;

int checkGPUStatus(){
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0){
        std::cerr << "No CUDA devices found\n";
        return 1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Global memory: " << prop.totalGlobalMem / B_TO_MB << " MB\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    size_t free_mem;
    size_t total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "Free memory: " << free_mem / B_TO_GB << "GB, total memory: " << total_mem / B_TO_GB << "GB\n";
    return 0;
}


struct weights{
    //all weights below
    __nv_bfloat16 *embeded_tokens;
    __nv_bfloat16 *input_layernorm[NUM_LAYERS];
    __nv_bfloat16 *mlp_down_proj[NUM_LAYERS];
    __nv_bfloat16 *mlp_gate_proj[NUM_LAYERS];
    __nv_bfloat16 *mlp_up_proj[NUM_LAYERS];
    __nv_bfloat16 *post_attn_layernorm[NUM_LAYERS];
    __nv_bfloat16 *self_attn_k_proj[NUM_LAYERS];
    __nv_bfloat16 *self_attn_o_proj[NUM_LAYERS];
    __nv_bfloat16 *self_attn_q_proj[NUM_LAYERS];
    __nv_bfloat16 *self_attn_v_proj[NUM_LAYERS];
    __nv_bfloat16 *norm;
};


//int loadWeightsSafetensors(int& addr){
int loadWeightsSafetensors(){

    std::ifstream fileModel("../model.safetensors", std::ios::binary);
    if (!fileModel.is_open()) {
        std::cerr << "Failed to open file: model.safetensors" << std::endl;
        return 1;
    }
    //header
    uint64_t header_size;
    fileModel.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
    std::string header;
    header.resize(header_size);
    fileModel.read(&header[0], header_size); 
    json header_json = json::parse(header);

    //weights - figuring out the layers (of tensors?)
    //alt ordered_json
    //unordered map? diff?
    std::map<std::string, std::pair<uint64_t, uint64_t>> tensor_map;
    uint64_t end_offset = 0;
    for(auto &[key, value]: header_json.items()){
        if(key == "__metadata__") continue;
        tensor_map[key] = {value["data_offsets"].at(0).get<uint64_t>(), value["data_offsets"].at(1).get<uint64_t>()};
        if(end_offset < value["data_offsets"].at(1).get<uint64_t>()){
            end_offset = value["data_offsets"].at(1).get<uint64_t>();
        }
    }

    char *weights_device;
    cudaMalloc(&weights_device, end_offset);
    
    char *weights_host = (char*)malloc(end_offset);
    fileModel.read(reinterpret_cast<char*>(weights_host), end_offset);

    cudaMemcpy(weights_device, weights_host, end_offset, cudaMemcpyHostToDevice);

    return 0;
}

int main(){
    loadWeightsSafetensors();

    
}
























