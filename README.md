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

Execution Configuration: 
<<<x, y:num of threads in thread block>>> syntax
y: blocks of threads are multiples of 32
<<<1, 256>>>
would run the computation once per thread
Modifying kernel:
threadIndex.x gives index of current thread within its block
blockDim.x contains number of threads in the block


basic1.cu: Runtime analysis (ns)
single thread: 128,665,708 
single block: 3,415,596 
multi block: 690,129 
+prefetching: 691,083 





