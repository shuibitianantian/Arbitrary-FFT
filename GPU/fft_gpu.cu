#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>
#include <stdio.h>
#include <omp.h>
#include "kernels.cuh"
#include <assert.h>
#include <stdlib.h>
#include <math.h>

// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>

// Decimation in Time FFT cuda version, used to perform dft of bluestein's algorithm
// Performs bit-reverse-copy before transform
// N must be power of 2
struct DitFFT_cuda {
    static constexpr char Name[] = "DitFFT_cuda";
    const std::size_t N;
    const int L;
    cuDoubleComplex* cuda_W;
    Comp* W;
    cuDoubleComplex* d_X;
    cuDoubleComplex* d_Y;
    size_t* d_I;
    std::size_t* I;

    DitFFT_cuda(std::size_t N) : N(N), L(__builtin_ctzll(N)) {
        using namespace std;
        // initialize W
        auto w = _2Pi / (Real) N;

        // allocate memory 
        cudaMalloc(&cuda_W, N*sizeof(cuDoubleComplex));
        cudaMalloc(&d_X, N*sizeof(cuDoubleComplex));
        cudaMalloc(&d_Y, N*sizeof(cuDoubleComplex));
        cudaMalloc(&d_I, N*sizeof(size_t));

        I = (std::size_t*) malloc(N*sizeof(std::size_t));
        W = (Comp*) malloc(N*sizeof(Comp));

        init_w_kernel<<<(N + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE>>>(cuda_W, w, N);

        // initialize I
        
        for (size_t i = 0, j = 0; i < N; ++i) {
            I[i] = j;
            for (size_t k = N >> 1; (j ^= k) < k; k >>= 1);
        }

        cudaMemcpy(d_I, I, N*sizeof(size_t), cudaMemcpyHostToDevice);

        for (size_t i = 0; i < N; ++i)
            W[i] = exp(Comp(0, w * (Real) i));
    }

    ~DitFFT_cuda(){
        free(W);
        free(I);
        cudaFree(d_X);
        cudaFree(d_Y);
        cudaFree(cuda_W);
        cudaFree(d_I);
    }

    void dft(Comp* Y, const Comp* X) const {
        using namespace std;
        cudaMemcpy(d_X, X, N*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Y, Y, N*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        
        bitrev_kernel<<<(N + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE>>>(d_Y, d_X, d_I, N);

        for(int i = 0; i < L; ++i){
            auto h = size_t{1} << i;
            dim3 blockDim(h < 1024 ? (BLOCKSIZE + h - 1) / h : 1, h < 1024 ? h : 1024);
            dim3 gridDim((N/(2*h) + blockDim.x - 1) / blockDim.x, (h + blockDim.y - 1) / blockDim.y);
            ditfft_kernel<<<gridDim, blockDim>>>(d_Y, cuda_W, i, N);
        }
        
        cudaMemcpy(Y, d_Y, N*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    }

    void idft(Comp* Y, const Comp* X) const {

        using namespace std;
        cudaMemcpy(d_X, X, N*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Y, Y, N*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        
        bitrev_kernel<<<(N + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE>>>(d_Y, d_X, d_I, N);

        double tt = omp_get_wtime();
        for(int i = 0; i < L; ++i){
            auto h = size_t{1} << i;
            dim3 blockDim(h < 1024 ? (BLOCKSIZE + h - 1) / h : 1, h < 1024 ? h : 1024);
            dim3 gridDim((N/(2*h) + blockDim.x - 1) / blockDim.x, (h + blockDim.y - 1) / blockDim.y);
            ditdifidft_kernel<<<gridDim, blockDim>>>(d_Y, cuda_W, i, N);
            Check_CUDA_Error("267");
        }

        complex_div_real<<<(N + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE>>>(d_Y, N);
        cudaMemcpy(Y, d_Y, N*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    }
};

// Forward transform uses DIF while inverse transform uses DIT, used to perform idft of bluestein's algorithm
// For convolution use, no bit-reverse-copy performed, performed in-place
// N must be power of 2
struct DifDitFFT_cuda {
    static constexpr char Name[] = "DifDitFFT_cuda";
    const std::size_t N;
    const int L;
    cuDoubleComplex* cuda_W;
    cuDoubleComplex* d_Y;
    Comp* W;

    DifDitFFT_cuda(std::size_t N) : N(N), L(__builtin_ctzll(N)) {
        using namespace std;
        auto w = _2Pi / (Real) N;

        cudaMalloc(&cuda_W, N*sizeof(cuDoubleComplex));
        cudaMalloc(&d_Y, N*sizeof(cuDoubleComplex));
        W = (Comp*) malloc(N*sizeof(Comp));

        init_w_kernel<<<(N + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE>>>(cuda_W, w, N);
        cudaDeviceSynchronize();
        cudaMemcpy(W, cuda_W, N*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    }

    ~DifDitFFT_cuda(){
        free(W);
        cudaFree(cuda_W);
        cudaFree(d_Y);
    }

    void dft(Comp* Y) {
        using namespace std;

        cudaMemcpy(d_Y, Y, N*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        double tt = omp_get_wtime();
        for(int i = L - 1; i >= 0; --i){
            auto h = size_t{1} << i;
            dim3 blockDim(h < 1024 ? (BLOCKSIZE + h - 1) / h : 1, h < 1024 ? h : 1024);
            dim3 gridDim((N/(2*h) + blockDim.x - 1) / blockDim.x, (h + blockDim.y - 1) / blockDim.y);
            ditdiffft_kernel<<<gridDim, blockDim>>>(d_Y, cuda_W, i, N);
        }

        cudaMemcpy(Y, d_Y, N*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    }

    void idft(Comp* Y) {
        using namespace std;

        cudaMemcpy(d_Y, Y, N*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

        double tt = omp_get_wtime();
        for(int i = 0; i < L; ++i){
            auto h = size_t{1} << i;
            dim3 blockDim(h < 1024 ? (BLOCKSIZE + h - 1) / h : 1, h < 1024 ? h : 1024);
            dim3 gridDim((N/(2*h) + blockDim.x - 1) / blockDim.x, (h + blockDim.y - 1) / blockDim.y);
            ditdifidft_kernel<<<gridDim, blockDim>>>(d_Y, cuda_W, i, N);
        }
        complex_div_real<<<(N + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE>>>(d_Y, N);
        cudaMemcpy(Y, d_Y, N*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    }
};


// cuda bluesteinFFT
struct BluesteinFFT_cuda {
    static constexpr char Name[] = "BluesteinFFT_cuda";
    const std::size_t N;
    const bool pow2;
    Comp* A; 
    Comp* B; 
    Comp* C;
    cuDoubleComplex* cuda_C;
    cuDoubleComplex* d_X;
    cuDoubleComplex* d_Y;
    cuDoubleComplex* d_A;
    cuDoubleComplex* d_B;
    double tt, run_time = 0;

    union {
        char c;
        DitFFT_cuda dit;
        DifDitFFT_cuda difdit;
    };

    BluesteinFFT_cuda(std::size_t N) : N(N), pow2(isPow2(N)) {
        using namespace std;
        if (pow2) {
            new(&dit) DitFFT_cuda(N);
            return;
        }

        new(&difdit) DifDitFFT_cuda(ceil2(N + N - 1));
        auto w = Pi / (Real) N;

        // allocate memory in GPU
        cudaMalloc(&d_X, N*sizeof(cuDoubleComplex));
        cudaMalloc(&d_Y, N*sizeof(cuDoubleComplex));
        cudaMalloc(&d_A, difdit.N*sizeof(cuDoubleComplex));
        cudaMalloc(&d_B, difdit.N*sizeof(cuDoubleComplex));
        cudaMalloc(&cuda_C, N*sizeof(cuDoubleComplex));

        // allocate memory for computation
        A = (Comp*) malloc(difdit.N*sizeof(Comp));
        B = (Comp*) malloc(difdit.N*sizeof(Comp));
        C = (Comp*) malloc(N*sizeof(Comp));
        
        // initialize C (W)
        init_C_kernel<<<(N + BLOCKSIZE - 1)/ BLOCKSIZE, BLOCKSIZE>>>(cuda_C, w, N);
        cudaMemcpy(C, cuda_C, N*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

        B[0] = C[0];

        for (size_t i = 1; i < N; ++i){
            B[i] = B[difdit.N - i] = C[i];
        }
        for(size_t i = N; i < difdit.N - N + 1; ++i){
            B[i] = Comp{};
        }
        difdit.dft(B);
        cudaMemcpy(d_B, B, difdit.N*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    }

    ~BluesteinFFT_cuda() {
        if (pow2) {
            dit.~DitFFT_cuda();
            return;
        }
        difdit.~DifDitFFT_cuda();
        cudaDeviceReset();
    }

    void dft(Comp* Y, const Comp* X) {
        if (pow2) {
            dit.dft(Y, X);
            return;
        }
        using namespace std;

        cudaMemcpy(d_X, X, N*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Y, Y, N*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

        
        tt = omp_get_wtime();
        complex_mul_conj_kernel<<<(N + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE>>>(d_A, d_X, cuda_C, N, difdit.N);

        // perform dft on A
        for(int i = difdit.L - 1; i >= 0; --i){
            auto h = size_t{1} << i;
            dim3 blockDim(h < 1024 ? (BLOCKSIZE + h - 1) / h : 1, h < 1024 ? h : 1024);
            dim3 gridDim((difdit.N/(2*h) + blockDim.x - 1) / blockDim.x, (h + blockDim.y - 1) / blockDim.y);
            ditdiffft_kernel<<<gridDim, blockDim>>>(d_A, difdit.cuda_W, i, difdit.N);
        }
        
        complex_self_mul_kernel<<<(difdit.N + BLOCKSIZE) / BLOCKSIZE, BLOCKSIZE>>>(d_A, d_B, difdit.N);
        
        // perform idft on A
        for(int i = 0; i < difdit.L; ++i){
            auto h = size_t{1} << i;
            dim3 blockDim(h < 1024 ? (BLOCKSIZE + h - 1) / h : 1, h < 1024 ? h : 1024);
            dim3 gridDim((difdit.N/(2*h) + blockDim.x - 1) / blockDim.x, (h + blockDim.y - 1) / blockDim.y);
            ditdifidft_kernel<<<gridDim, blockDim>>>(d_A, difdit.cuda_W, i, difdit.N);
        }
        
        complex_div_real_mul_conj<<<(difdit.N + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE>>>(d_A, d_Y, cuda_C, difdit.N, N);

        cudaDeviceSynchronize();
        
        run_time += omp_get_wtime() - tt;
        tt = omp_get_wtime();

        cudaMemcpy(Y, d_Y, N*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    }

    void idft(Comp* Y, const Comp* X) {
        if (pow2) {
            dit.idft(Y, X);
            return;
        }
        using namespace std;

        cudaMemcpy(d_X, X, N*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Y, Y, N*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);


        complex_mul_kernel<<<(N + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE>>>(d_A, d_X, cuda_C, N, difdit.N);
    
        dim3 blockDim(32, 32);
        for(int i = difdit.L - 1; i >= 0; --i){
            auto h = size_t{1} << i;
            dim3 gridDim((difdit.N/(2*h) + blockDim.x - 1) / blockDim.x, (h + blockDim.x - 1) / blockDim.x);
            ditdiffft_kernel<<<gridDim, blockDim>>>(d_A, difdit.cuda_W, i, difdit.N);
        }

        complex_self_conj_mul_kernel<<<(difdit.N + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE>>>(d_A, d_B, difdit.N);

        for(int i = 0; i < difdit.L; ++i){
            auto h = size_t{1} << i;
            dim3 gridDim((difdit.N/(2*h) + blockDim.x - 1) / blockDim.x, (h + blockDim.x - 1) / blockDim.x);
            ditdifidft_kernel<<<gridDim, blockDim>>>(d_A, difdit.cuda_W, i, difdit.N);
        }
        complex_div_real_mul<<<(difdit.N + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE>>>(d_A, d_Y, cuda_C, difdit.N, N);
        cudaMemcpy(Y, d_Y, N*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    }
};

/*
    Function to run fft and report the time
    
    Params:
    ~ std::size_t sign: 1 represents forward dft, 0 represents inverse dft (idft)
    ~ std::size_t N: size of input data
    ~ Comp* input: input data
*/
void fft_cuda(std::size_t sign, std::size_t N, Comp* input, Comp* output){
    using namespace std;
    BluesteinFFT_cuda bluestein_cuda(N);

    if(sign == 1){
        bluestein_cuda.dft(output, input);
    }else{
        bluestein_cuda.idft(output, input);
    }
    cout << "Blustein cuda Run time: " << bluestein_cuda.run_time << endl;
}

/*
    Function to run benchmark (cufft)
    
    Params:
    ~ Comp* Data: input data
    ~ int LENGTH: length of input data
*/
void benchmark(Comp* Data, int LENGTH){
    cufftDoubleComplex* CompData = (cufftDoubleComplex*) malloc(LENGTH*sizeof(cufftDoubleComplex));
    for(int i = 0; i < LENGTH; ++i){
        CompData[i].x = real(Data[i]);
        CompData[i].y = imag(Data[i]);
    }

    cufftDoubleComplex *d_fftData;
    cudaMalloc((void**)&d_fftData, LENGTH * sizeof(cufftDoubleComplex));// allocate memory for the data in device
    cudaMemcpy(d_fftData, CompData, LENGTH * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);// copy data from host to device

    cufftHandle plan;// cuda library function handle
    cufftPlan1d(&plan, LENGTH, CUFFT_Z2Z, 1);//declaration
    
    double tt = omp_get_wtime();
    cufftExecZ2Z(plan, (cufftDoubleComplex*)d_fftData, (cufftDoubleComplex*)d_fftData, CUFFT_FORWARD);//execute
    cudaDeviceSynchronize();//wait to be done
    std::cout << "Bechmark run time: " << omp_get_wtime() - tt << std::endl;

    cudaMemcpy(CompData, d_fftData, LENGTH * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);// copy the result from device to host

    Check_CUDA_Error("error");

    cufftDestroy(plan);
    cudaFree(d_fftData);
    for(int i = 0; i < LENGTH; ++i)
        Data[i] = Comp(CompData[i].x, CompData[i].y);

    free(CompData);
    
}