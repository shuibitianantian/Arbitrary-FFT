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

// Helper functions for CUDA
#include "device_launch_parameters.h"

struct NaiveDFT_cuda {
    static constexpr char Name[] = "NaiveDFT_cuda";
    const std::size_t N;

    NaiveDFT_cuda(std::size_t N) : N(N) {}

    ~NaiveDFT_cuda(){cudaDeviceReset();}

    void dft(Comp* Y, const Comp* X){
      using namespace std;

      size_t xblock = std::sqrt(BLOCKSIZE); // currently support 2^n
      dim3 dimBlock(xblock, xblock); // define the size of block
      
      size_t xgrid = (N + dimBlock.x - 1) / dimBlock.x;
      size_t ygrid = (N + dimBlock.y - 1) / dimBlock.y;
      dim3 dimGrid(xgrid, ygrid); 

      // initialize host and device of X
      cuDoubleComplex* h_X = (cuDoubleComplex*) malloc(N*sizeof(cuDoubleComplex));  // allocate memory to host
      Comp_to_cuComp(X, h_X, N);  // convert Comp to cuda complex
      cuDoubleComplex* d_X;
      cudaMalloc(&d_X, N*sizeof(cuDoubleComplex));
      cudaMemcpy(d_X, h_X, N*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

      // initialize host and device of Y
      cuDoubleComplex* h_Y = (cuDoubleComplex*) malloc(N*ygrid*sizeof(cuDoubleComplex));
      cuDoubleComplex* d_Y;
      cudaMalloc(&d_Y, N*ygrid*sizeof(cuDoubleComplex));
      for(int i = 0; i < N*ygrid; ++i)
        h_Y[i] = make_cuDoubleComplex(0.0, 0.0);
      cudaMemcpy(d_Y, h_Y, N*ygrid*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
      cudaDeviceSynchronize();

      double tt = omp_get_wtime();
      dft_kernel2d<<<dimGrid, dimBlock>>>(d_X, d_Y, N, ygrid);
      Check_CUDA_Error("Error");
      reduction_2d<<<(N + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE>>>(d_Y, N, ygrid);
      Check_CUDA_Error("Error");
      cudaDeviceSynchronize();
      cudaMemcpy(h_Y, d_Y, N*ygrid*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

      for(int i = 0; i < N; ++i)
        Y[i] = Comp(cuCreal(h_Y[i*ygrid]), cuCimag(h_Y[i*ygrid]));

        // cout << "[" << Name  << "] (dft) run time: " <<  omp_get_wtime() - tt << endl;
      free(h_Y);
      free(h_X);
    }

    void idft(Comp* Y, const Comp* X) {
      using namespace std;

      size_t xblock = std::sqrt(BLOCKSIZE); // currently support 2^n
      dim3 dimBlock(xblock, xblock); // define the size of block
      
      size_t xgrid = (N + dimBlock.x - 1) / dimBlock.x;
      size_t ygrid = (N + dimBlock.y - 1) / dimBlock.y;
      dim3 dimGrid(xgrid, ygrid); 

      // initialize host and device of X
      cuDoubleComplex* h_X = (cuDoubleComplex*) malloc(N*sizeof(cuDoubleComplex));  // allocate memory to host
      Comp_to_cuComp(X, h_X, N);  // convert Comp to cuda complex
      cuDoubleComplex* d_X;
      cudaMalloc(&d_X, N*sizeof(cuDoubleComplex));
      cudaMemcpy(d_X, h_X, N*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

      // initialize host and device of Y
      cuDoubleComplex* h_Y = (cuDoubleComplex*) malloc(N*ygrid*sizeof(cuDoubleComplex));
      cuDoubleComplex* d_Y;
      cudaMalloc(&d_Y, N*ygrid*sizeof(cuDoubleComplex));
      for(int i = 0; i < N*ygrid; ++i)
        h_Y[i] = make_cuDoubleComplex(0.0, 0.0);
      cudaMemcpy(d_Y, h_Y, N*ygrid*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
      cudaDeviceSynchronize();

      double tt = omp_get_wtime();
      idft_kernel2d<<<dimGrid, dimBlock>>>(d_X, d_Y, N, ygrid);
      Check_CUDA_Error("Error");
      reduction_2d<<<(N + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE>>>(d_Y, N, ygrid);
      Check_CUDA_Error("Error");
      cudaDeviceSynchronize();
      cudaMemcpy(h_Y, d_Y, N*ygrid*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
  
      for(int i = 0; i < N; ++i)
        Y[i] = Comp(cuCreal(h_Y[i*ygrid]) / N, cuCimag(h_Y[i*ygrid]) / N);
    cout << "[" << Name  << "] (idft) run time: " <<  omp_get_wtime() - tt << endl;

      free(h_Y);
      free(h_X);
    }
};

// Decimation in Time FFT
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

        // cudaMemcpy(W, cuda_W, N*sizeof(size_t), cudaMemcpyDeviceToHost);
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


// Forward transform uses DIF while inverse transform uses DIT
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
        Check_CUDA_Error("395");
        // cout << "[" << Name  << "] (dft) run time: " <<  omp_get_wtime() - tt << endl;
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
        Check_CUDA_Error("406");
        // cout << "[" << Name  << "] (idft) run time: " <<  omp_get_wtime() - tt << endl;
    }
};

template<class I>
constexpr I ceil2(I x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    if (sizeof(I) > 1)
        x |= x >> 8;
    if (sizeof(I) > 2)
        x |= x >> 16;
    if (sizeof(I) > 4)
        x |= x >> 32;
    return x + 1;
}

template<class I>
constexpr bool isPow2(I x) { return ((x ^ (x - 1)) >> 1) == x - 1; }

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

Real error(const Comp* A, const Comp* B, std::size_t N) {
    using namespace std;
    Real res = 0;
    for (size_t i = 0; i < N; ++i)
        res = max(res, abs(A[i] - B[i]));
    return res;
}

std::mt19937_64 R(std::random_device{}());

Real randReal(Real lo, Real up) {
    return std::uniform_real_distribution<Real>(lo, up)(R);
}

Comp randComp(Real lo, Real up) {
    return {randReal(lo, up), randReal(lo, up)};
}

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

void print_vector(Comp* x, int N){
    for(int i = 0; i < N; ++i)
        std::cout << x[i] << std::endl;
    std::cout << std::endl;
}

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


int main(int argc, char* argv[]){
    using namespace std;
    int N = atoi(argv[1]);
    
    Comp* input = (Comp*) malloc(N*sizeof(Comp));
    Comp* output = (Comp*) malloc(N*sizeof(Comp));
    Comp* original = (Comp*) malloc(N*sizeof(Comp));

    for(int i = 0; i < N; ++i)
        original[i] = input[i] = randComp(-10, 10);

    NaiveDFT_cuda naive_cuda(N);
    fft_cuda(1, N, input, output);
    
    benchmark(input, N);
    cout << "Error: " << error(output, input, N) << endl;

    free(input);
    free(output);
    free(original);

    return 0;
}



