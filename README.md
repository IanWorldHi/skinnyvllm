# skinnyvllm + ml CUDA learning

## Notes:

Safetenors format:
header size (8 bytes unsigned)
header
tensors data

Compile with nvcc 
NSight Systems CLI: Can use to profile it (find runtime)
nsys profile -t cuda --stats=true ./exec
- some additional flags for more detailed reports/traces
- cudaDeviceSyncrhonize: host blockign till gpu time, basically gpu time
- cudaMallocManaged: roughly cpu time











