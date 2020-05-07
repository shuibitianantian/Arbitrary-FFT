Parallel FFT using CUDA
==========================

## Disclaimer
* All information provided about NYU Prince cluster in this file is up to date
  as of May 6 2020.
* The results related to CUDA used in the presentation was out dated. Those
  results are regenerated in this commit.
* The results used in the presentation was the output of program ran at RTX 2060.

## Final Version
* `fft_gpu.cu`: The final CUDA implementation of this project
  (using Bluestein's algorithm).

* `run_benchmark.cu`: Program to test our implementation of fft and compare with Nvidia cufft.

## Build and Run on NYU Prince Cluster
1. `$ cd jobs`
2. `$ sbatch run.s`: Automatically running test on number from 10 to 1e7 and prime number 1e1+1, 1e2+1, 1e3+9, 1e4+7, 1e5+3, 1e6+3, 1e7+19.

## Build and Run on Your Own Computer
* On Linux: `nvcc -std=c++11 -o main ./run_benchmark.cu -lcufft  -Xcompiler -fopenmp`
* On windows: `nvcc -o main .\test_benchmark.cu -I "Path\to\CUDA\version\includes" -L"Path\to\CUDA\version\lib" -lcufft`.

## Source Files
* `utils.cuh`: includes helper functions, for example function ` std::Complex<double> randComp(double lo, double up)` used to generate random complex number for testing.

* `kernels.cuh`: includes all kernel functions used in gpu, the most important kernels are `ditfft_kernel` and `ditdiffft_kernel`.

* `fft_cuda.cu`: includes CUDA implementation of bluestein's algorithm, Decimation in time algorithm.

## Using Function
1. `void fft_gpu::fft_cuda(std::size_t sign, std::size_t N, Comp* input, Comp* output)` is the entry point of the tool, in order to use this function, please put `#include "fft_cuda.cu"` in your source file.

2. You can compile `test.cu` and run it. If you do not want to ouput the running time, just modified the code of `fft_gpu::fft_cuda`.

3. Current implementation can not deal with data length more than 1e7
