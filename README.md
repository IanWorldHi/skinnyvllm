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
Clock cycle/Clock Rate
Thread generation and scheduling
How does the CUDA underlying library launch a grid of threads, (i'm assuming this is difficult to implement?)
Dynamic random access memory? In terms of hardware whare are the diff types of memory and their use ie like vram dram as well 
oh shoot i think codamemcpy was cahnged pretty recently, 4th ed was pretty recent
Does buying it+the ebook give me the answer sheet to the exercises as well?



Explained later:
Global memeory? it's called that for device
Clarification on DRAM, HBM, HBM2 and their specifics (check if I have redundent references in my notes)
What is considered as execution resource?
Does executing an application slowly take less power over the time taken to execute it? (notes under Transparent Scalability)
What is coined as an execution unit?
What is a core in relation to the number of threads/warsp/breakdown of blocks?
Pascal architecture again?
For conditional divergence, how/why do the threads reconverge after the conditional? Are they barrier synchronzied? Can they not run in parallel? Something to do with independent thread scheduling?
    Or is it that independent thread scheduling largely mitages the conditional divergence loss of runtime?
        wtf is it? "Threads in a warp are
        executed following the SIMD model. If threads in the same warp diverge by taking dif-
        ferent execution paths, the processing block executes these paths in passes in which
        each thread is active only in the pass corresponding to the path that it takes."
        (Unrelated) Can you mute a desktop app windwos? 
        For resource partitioning and occupency, how come SMs can have different number of blocks but are also limited in blocks? Isn't a block just an efficent encapsulation for threads?
Dynamic partitioning registers of an SM? For blocks as well as for registers per thread? How does the maximum limit work then if it is dynamic?
Chapter 5, what is shared memory in conjunction to registers and being on-chip? Is it just global memory?
Figure 5.2, what type of CUDA memory is not included in the textbook that is important???? Why would I not be learning it?
Where does the processor chip fit into the whole thing?
What is an automatic variable? Why is it different if it is an array?
How efficent is constnat memory in comparision? Also what is caching, how does it work hardwarewise in comparision to plain memory access?
How does the alg determine what is frequently accessed for global memory to be loaded into shared memory without adding additionaly unnecessarily burden/computation?
Atomic operations?
Do you have to use pointers to gain access to global memory?




CUDA C Programming Guide???



My device:
SMs: 26
MaxThreadsPerBlock: 1024








