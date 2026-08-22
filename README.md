# skinnyvllm + ml CUDA learning

## Notes:

Reading through Programming Massively Parallel Processors 4th-5th ed
- prob should change the git proj name
- 4th is a bit oudated, ie) cudaMemPrefetchAsync
- is cuda now fully C++? or at least not C?

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
would run the computation once per thread (one block)
Modifying kernel:
threadIndex.x gives index of current thread within its block
blockDim.x contains number of threads in the block

Shorthand:
_d for device code
_h for host code

basic1.cu: Runtime analysis (ns)
single thread: 128,665,708 
single block: 3,415,596 
multi block: 690,129 
+prefetching: 691,083 


Questions for wifi:
Why is it called serial code for CPU serial code? 
Clock cycle
Thread generation and scheduling
What does it mean by a thread is sequential? shouldn't it be by default, isn't that the only option?
How does the CUDA underlying library launch a grid of threads, (i'm assuming this is difficult to implement?)
Dynamic random access memory? In terms of hardware whare are the diff types of memory and their use ie like vram dram as well
for cudaMalloc, do I have to cast the pointer to void (generic pointer)? it returns a generic obj?
Oh cudamalloc vs cudamallocmanaged?
What does kernel mean outside of this context, ie a linux kernel? 
oh shoot i think codamemcpy was cahnged pretty recently, 4th ed was pretty recent
I wonder what the average salary in china is, vs what the higher end gov managerial ones are and what the avg living cost is
Wait can I call malloc on the same thing twice


Explained later:
Global memeory? it's called that for device










