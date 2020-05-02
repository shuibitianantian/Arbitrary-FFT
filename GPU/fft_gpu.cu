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

// Naive implementation of DFT
// Arbitrary N
struct NaiveDFT {
    static constexpr char Name[] = "NaiveDFT";
    const std::size_t N;

    NaiveDFT(std::size_t N) : N(N) {}

    void dft(Comp* Y, const Comp* X) const {
        using namespace std;
        double tt = omp_get_wtime();
        for (size_t k = 0; k < N; ++k) {
            Comp y = 0;
            for (size_t n = 0; n < N; ++n)
                y += exp(Comp(0, -_2Pi / (Real) N * (Real) k * (Real) n)) * X[n];
            Y[k] = y;
        }
        cout << "[" << Name  << "] (dft) run time: " <<  omp_get_wtime() - tt << endl;
    }

    void idft(Comp* Y, const Comp* X) const {
        using namespace std;
        for (size_t k = 0; k < N; ++k) {
            Comp y = 0;
            for (size_t n = 0; n < N; ++n)
                y += exp(Comp(0, _2Pi / (Real) N * (Real) k * (Real) n)) * X[n];
            Y[k] = y / (Real) N;
        }
    }
};

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

        cout << "[" << Name  << "] (dft) run time: " <<  omp_get_wtime() - tt << endl;
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
    cout << "[" << Name  << "] (dft) run time: " <<  omp_get_wtime() - tt << endl;

      free(h_Y);
      free(h_X);
    }
};

// Decimation in Time FFT
// Performs bit-reverse-copy before transform
// N must be power of 2
struct DitFFT {
    static constexpr char Name[] = "DitFFT";
    const std::size_t N;
    const int L;
    std::vector<Comp> W;
    std::vector<std::size_t> I;

    DitFFT(std::size_t N) : N(N), L(__builtin_ctzll(N)), W(N), I(N) {
        using namespace std;
        auto w = _2Pi / (Real) N;
        for (size_t i = 0; i < N; ++i)
            W[i] = exp(Comp(0, w * (Real) i));
        for (size_t i = 0, j = 0; i < N; ++i) {
            I[i] = j;
            for (size_t k = N >> 1; (j ^= k) < k; k >>= 1);
        }
    }

    void dft(Comp* Y, const Comp* X) const {
        using namespace std;
        double tt = omp_get_wtime();
        bitrev(Y, X);
        for (int i = 0; i < L; ++i) {
            auto g = N >> (i + 1);
            auto h = size_t{1} << i;
            for (size_t j = 0; j < N; j += h << 1)
                for (size_t k = 0; k < h; ++k) {
                    auto u = Y[j + k];
                    auto v = conj(W[k * g]) * Y[j + k + h];
                    Y[j + k] = u + v;
                    Y[j + k + h] = u - v;;
                }
        }
        cout << "[" << Name  << "] (dft) run time: " <<  omp_get_wtime() - tt << endl;
    }

    void idft(Comp* Y, const Comp* X) const {
        using namespace std;
        double tt = omp_get_wtime();
        bitrev(Y, X);
        for (int i = 0; i < L; ++i) {
            auto g = N >> (i + 1);
            auto h = size_t{1} << i;
            for (size_t j = 0; j < N; j += h << 1)
                for (size_t k = 0; k < h; ++k) {
                    auto u = Y[j + k];
                    auto v = W[k * g] * Y[j + k + h];
                    Y[j + k] = u + v;
                    Y[j + k + h] = u - v;;
                }
        }
        for (size_t i = 0; i < N; ++i)
            Y[i] /= (Real) N;
        cout << "[" << Name  << "] (dft) run time: " <<  omp_get_wtime() - tt << endl;

    }

    void bitrev(Comp* Y, const Comp* X) const {
        using namespace std;
        for (size_t i = 0; i < N; ++i)
            Y[i] = X[I[i]];
    }
};

// cuda ditfft
struct DitFFT_cuda {
    static constexpr char Name[] = "DitFFT_cuda";
    const std::size_t N;
    const int L;
    cuDoubleComplex* cuda_W;
    Comp* W;
    std::size_t* I;

    DitFFT_cuda(std::size_t N) : N(N), L(__builtin_ctzll(N)) {
        using namespace std;
        // initialize W
        auto w = _2Pi / (Real) N;
        cudaMalloc(&cuda_W, N*sizeof(cuDoubleComplex));
        init_w_kernel<<<(N + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE>>>(cuda_W, w, N);
        cudaDeviceSynchronize();
        
        // initialize I
        I = (std::size_t*) malloc(N*sizeof(std::size_t));
        for (size_t i = 0, j = 0; i < N; ++i) {
            I[i] = j;
            for (size_t k = N >> 1; (j ^= k) < k; k >>= 1);
        }

        W = (Comp*) malloc(N*sizeof(Comp));
        for (size_t i = 0; i < N; ++i)
            W[i] = exp(Comp(0, w * (Real) i));
    }

    ~DitFFT_cuda(){
        free(W);
        free(I);
        cudaDeviceReset();
    }

    void dft(Comp* Y, const Comp* X) const {
        using namespace std;

        // cuDoubleComplex* d_W;
        // cudaMalloc(&d_W, N*sizeof(cuDoubleComplex));
        // cudaMemcpy(d_W, cuda_W, N*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

        cuDoubleComplex* h_X = (cuDoubleComplex*) malloc(N*sizeof(cuDoubleComplex));
        cuDoubleComplex* h_Y = (cuDoubleComplex*) malloc(N*sizeof(cuDoubleComplex));
        Comp_to_cuComp(X, h_X, N);
        Comp_to_cuComp(Y, h_Y, N);

        cuDoubleComplex* d_X;
        cuDoubleComplex* d_Y;
        size_t* d_I;

        cudaMalloc(&d_X, N*sizeof(cuDoubleComplex));
        cudaMalloc(&d_Y, N*sizeof(cuDoubleComplex));
        cudaMalloc(&d_I, N*sizeof(size_t));

        cudaMemcpy(d_X, h_X, N*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Y, h_Y, N*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        cudaMemcpy(d_I, I, N*sizeof(size_t), cudaMemcpyHostToDevice);
        
        
        bitrev_kernel<<<(N + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE>>>(d_Y, d_X, d_I, N);
        cudaMemcpy(h_Y, d_Y,  N*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
        // cuComp_to_Comp(h_Y, Y, N);
        double tt = omp_get_wtime();
        for(int i = 0; i < L; ++i){
            auto h = size_t{1} << i;
            dim3 blockDim(32, 32);
            dim3 gridDim((N/(2*h) + blockDim.x - 1) / blockDim.x, (h + blockDim.x - 1) / blockDim.x);
            ditfft_kernel<<<gridDim, blockDim>>>(d_Y, cuda_W, i, N);
            Check_CUDA_Error("Error.");
        }
        cudaMemcpy(h_Y, d_Y, N*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
        cout << "[" << Name  << "] (dft) run time: " <<  omp_get_wtime() - tt << endl;
        cuComp_to_Comp(h_Y, Y, N);
    }

    void idft(Comp* Y, const Comp* X) const {
        using namespace std;
        double tt = omp_get_wtime();
        bitrev(Y, X);
        for (int i = 0; i < L; ++i) {
            auto g = N >> (i + 1);
            auto h = size_t{1} << i;
            for (size_t j = 0; j < N; j += h << 1)
                for (size_t k = 0; k < h; ++k) {
                    auto u = Y[j + k];
                    auto v = W[k * g] * Y[j + k + h];
                    Y[j + k] = u + v;
                    Y[j + k + h] = u - v;;
                }
        }
        for (size_t i = 0; i < N; ++i)
            Y[i] /= (Real) N;
        cout << "[" << Name  << "] (dft) run time: " <<  omp_get_wtime() - tt << endl;

    }

    void bitrev(Comp* Y, const Comp* X) const {
        using namespace std;
        for (size_t i = 0; i < N; ++i)
            Y[i] = X[I[i]];
    }
};

// Decimation in Frequency FFT
// Performs bit-reverse-copy after transform
// N must be power of 2
struct DifFFT {
    static constexpr char Name[] = "DifFFT";
    const std::size_t N;
    const int L;
    std::vector<Comp> W;
    std::vector<std::pair<std::size_t, std::size_t>> I;

    DifFFT(std::size_t N) : N(N), L(__builtin_ctzll(N)), W(N) {
        using namespace std;
        auto w = _2Pi / (Real) N;
        for (size_t i = 0; i < N; ++i)
            W[i] = exp(Comp(0, w * (Real) i));
        for (size_t i = 0, j = 0; i < N; ++i) {
            if (i < j)
                I.emplace_back(i, j);
            for (size_t k = N >> 1; (j ^= k) < k; k >>= 1);
        }
    }

    void dft(Comp* Y, const Comp* X) const {
        using namespace std;
        auto h = N >> 1;
        for (size_t k = 0; k < h; ++k) {
            auto u = X[k];
            auto v = X[k + h];
            Y[k] = u + v;
            Y[k + h] = conj(W[k]) * (u - v);
        }
        for (int i = L - 2; i >= 0; --i) {
            auto g = N >> (i + 1);
            h = size_t{1} << i;
            for (size_t j = 0; j < N; j += h << 1)
                for (size_t k = 0; k < h; ++k) {
                    auto u = Y[j + k];
                    auto v = Y[j + k + h];
                    Y[j + k] = u + v;
                    Y[j + k + h] = conj(W[k * g]) * (u - v);
                }
        }
        bitrev(Y);
    }

    void idft(Comp* Y, const Comp* X) const {
        using namespace std;
        auto h = N >> 1;
        for (size_t k = 0; k < h; ++k) {
            auto u = X[k];
            auto v = X[k + h];
            Y[k] = u + v;
            Y[k + h] = W[k] * (u - v);
        }
        for (int i = L - 2; i >= 0; --i) {
            auto g = N >> (i + 1);
            h = size_t{1} << i;
            for (size_t j = 0; j < N; j += h << 1)
                for (size_t k = 0; k < h; ++k) {
                    auto u = Y[j + k];
                    auto v = Y[j + k + h];
                    Y[j + k] = u + v;
                    Y[j + k + h] = W[k * g] * (u - v);
                }
        }
        for (size_t i = 0; i < N; ++i)
            Y[i] /= (Real) N;
        bitrev(Y);
    }

    void bitrev(Comp* Y) const {
        using namespace std;
        for (auto& p : I)
            swap(Y[p.first], Y[p.second]);
    }
};

// Forward transform uses DIF while inverse transform uses DIT
// For convolution use, no bit-reverse-copy performed, performed in-place
// N must be power of 2
struct DifDitFFT {
    static constexpr char Name[] = "DifDitFFT";
    const std::size_t N;
    const int L;
    std::vector<Comp> W;

    DifDitFFT(std::size_t N) : N(N), L(__builtin_ctzll(N)), W(N) {
        using namespace std;
        auto w = _2Pi / (Real) N;
        for (size_t i = 0; i < N; ++i)
            W[i] = exp(Comp(0, w * (Real) i));
    }

    void dft(Comp* Y) const {
        using namespace std;
        double tt = omp_get_wtime();
        for (int i = L - 1; i >= 0; --i) {
            auto g = N >> (i + 1);
            auto h = size_t{1} << i;
            for (size_t j = 0; j < N; j += h << 1)
                for (size_t k = 0; k < h; ++k) {
                    auto u = Y[j + k];
                    auto v = Y[j + k + h];
                    Y[j + k] = u + v;
                    Y[j + k + h] = conj(W[k * g]) * (u - v);
                }
        }
        cout << "[" << Name  << "] (dft) run time: " <<  omp_get_wtime() - tt << endl;
    }

    void idft(Comp* Y) const {
        using namespace std;
        double tt = omp_get_wtime();
        for (int i = 0; i < L; ++i) {
            auto g = N >> (i + 1);
            auto h = size_t{1} << i;
            for (size_t j = 0; j < N; j += h << 1)
                for (size_t k = 0; k < h; ++k) {
                    auto u = Y[j + k];
                    auto v = W[k * g] * Y[j + k + h];
                    Y[j + k] = u + v;
                    Y[j + k + h] = u - v;;
                }
        }
        for (size_t i = 0; i < N; ++i)
            Y[i] /= (Real) N;

        cout << "[" << Name  << "] (idft) run time: " <<  omp_get_wtime() - tt << endl;

    }
};

struct DifDitFFT_cuda {
    static constexpr char Name[] = "DifDitFFT_cuda";
    const std::size_t N;
    const int L;
    cuDoubleComplex* cuda_W;
    Comp* W;

    DifDitFFT_cuda(std::size_t N) : N(N), L(__builtin_ctzll(N)) {
        using namespace std;
        auto w = _2Pi / (Real) N;
        cudaMalloc(&cuda_W, N*sizeof(cuDoubleComplex));
        init_w_kernel<<<(N + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE>>>(cuda_W, w, N);
        cudaDeviceSynchronize();

        W = (Comp*) malloc(N*sizeof(Comp));
        for (size_t i = 0; i < N; ++i)
            W[i] = exp(Comp(0, w * (Real) i));
    }

    ~DifDitFFT_cuda(){cudaDeviceReset();}

    void dft(Comp* Y) {
        using namespace std;
        cuDoubleComplex* h_Y = (cuDoubleComplex*) malloc(N*sizeof(cuDoubleComplex));
        Comp_to_cuComp(Y, h_Y, N);

        cuDoubleComplex* d_Y;
        cudaMalloc(&d_Y, N*sizeof(cuDoubleComplex));
        cudaMemcpy(d_Y, h_Y, N*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

        double tt = omp_get_wtime();
        for(int i = L - 1; i >= 0; --i){
            auto h = size_t{1} << i;
            dim3 blockDim(32, 32);
            dim3 gridDim((N/(2*h) + blockDim.x - 1) / blockDim.x, (h + blockDim.x - 1) / blockDim.x);
            ditdiffft_kernel<<<gridDim, blockDim>>>(d_Y, cuda_W, i, N);
            Check_CUDA_Error("475.");
        }

        cudaMemcpy(h_Y, d_Y, N*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
        cout << "[" << Name  << "] (dft) run time: " <<  omp_get_wtime() - tt << endl;

        cuComp_to_Comp(h_Y, Y, N);
        free(h_Y);
        cudaFree(d_Y);
    }

    void idft(Comp* Y) {
        using namespace std;
        cuDoubleComplex* d_Y;
        cuDoubleComplex* h_Y = (cuDoubleComplex*) malloc(N*sizeof(cuDoubleComplex));
        
        Comp_to_cuComp(Y, h_Y, N);
        cudaMalloc(&d_Y, N*sizeof(cuDoubleComplex));
        cudaMemcpy(d_Y, h_Y, N*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

        double tt = omp_get_wtime();
        for(int i = 0; i < L; ++i){
            auto h = size_t{1} << i;
            dim3 blockDim(32, 32);
            dim3 gridDim((N/(2*h) + blockDim.x - 1) / blockDim.x, (h + blockDim.x - 1) / blockDim.x);
            ditdifidft_kernel<<<gridDim, blockDim>>>(d_Y, cuda_W, i, N);
            Check_CUDA_Error("500");
        }
        
        cudaMemcpy(h_Y, d_Y, N*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
        cout << "[" << Name  << "] (idft) run time: " <<  omp_get_wtime() - tt << endl;
        cudaDeviceSynchronize();

        cuComp_to_Comp(h_Y, Y, N);

        for (size_t i = 0; i < N; ++i)
            Y[i] /= (Real) N;

        free(h_Y);
        cudaFree(d_Y);
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

// Bluestein's Algorithm: Chirp Z-Transform
// Arbitrary N
struct BluesteinFFT {
    static constexpr char Name[] = "BluesteinFFT";
    const std::size_t N;
    const bool pow2;
    std::vector<Comp> A, B, C;
    union {
        char c;
        DitFFT dit;
        DifDitFFT difdit;
    };

    BluesteinFFT(std::size_t N) : N(N), pow2(isPow2(N)) {
        if (pow2) {
            new(&dit) DitFFT(N);
            return;
        }
        new(&difdit) DifDitFFT(ceil2(N + N - 1));
        using namespace std;
        A.resize(difdit.N);
        B.resize(difdit.N);
        C.resize(N);
        auto w = Pi / (Real) N;
        for (size_t i = 0; i < N; ++i)
            C[i] = exp(Comp(0, w * (Real) (i * i)));
        B[0] = C[0];
        for (size_t i = 1; i < N; ++i)
            B[i] = B[difdit.N - i] = C[i];
        difdit.dft(B.data());
    }

    ~BluesteinFFT() {
        if (pow2) {
            dit.~DitFFT();
            return;
        }
        difdit.~DifDitFFT();
    }

    void dft(Comp* Y, const Comp* X) {
        if (pow2) {
            dit.dft(Y, X);
            return;
        }
        using namespace std;
        for (size_t i = 0; i < N; ++i)
            A[i] = X[i] * conj(C[i]);
        fill(A.begin() + (ptrdiff_t) N, A.end(), Comp{});
        difdit.dft(A.data());
        for (size_t i = 0; i < difdit.N; ++i)
            A[i] *= B[i];
        difdit.idft(A.data());
        for (size_t i = 0; i < N; ++i)
            Y[i] = A[i] * conj(C[i]);
    }

    void idft(Comp* Y, const Comp* X) {
        if (pow2) {
            dit.idft(Y, X);
            return;
        }
        using namespace std;
        for (size_t i = 0; i < N; ++i)
            A[i] = X[i] * C[i];
        fill(A.begin() + (ptrdiff_t) N, A.end(), Comp{});
        difdit.dft(A.data());
        for (size_t i = 0; i < difdit.N; ++i)
            A[i] *= conj(B[i]);
        difdit.idft(A.data());
        for (size_t i = 0; i < N; ++i)
            Y[i] = A[i] * C[i] / (Real) N;
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

template<class Tr>
void testIdentity(Tr&& T) {
    using namespace std;
    auto N = T.N;
    vector<Comp> X(N), Y(N), Z(N);
    for (auto& x : X)
        x = randComp(-100, 100);
    T.dft(Y.data(), X.data());
    T.idft(Z.data(), Y.data());
    auto err = error(X.data(), Z.data(), N);
    cout << "[" << T.Name << "] err(x, idft(dft(x))) = " << err << endl;
}

template<class Tr>
void testConvolution(Tr&& T) {
    using namespace std;
    auto N = T.N;
    auto M = N / 2;
    vector<Comp> A(N), B(N), C(N), D(N);
    vector<Comp> Af(N), Bf(N), Df(N);
    for (size_t i = 0; i < M; ++i) {
        A[i] = randComp(-100, 100);
        B[i] = randComp(-100, 100);
    }
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < M; ++j)
            C[i + j] += A[i] * B[j];
    T.dft(Af.data(), A.data());
    T.dft(Bf.data(), B.data());
    for (size_t i = 0; i < N; ++i)
        Df[i] = Af[i] * Bf[i];
    T.idft(D.data(), Df.data());
    auto err = error(C.data(), D.data(), M * 2 - 1);
    cout << "[" << T.Name << "] err(a*b, idft(dft(a)*dft(b))) = " << err << endl;
}

template<class Tr>
void testAll(Tr&& T) {
    testIdentity(std::forward<Tr>(T));
    // testConvolution(std::forward<Tr>(T));
}

template<class... F>
void testCmp(const NaiveDFT& naive, F&&... ffts) {
    using namespace std;
    auto N = naive.N;
    vector<Comp> A(N);
    for (auto& a : A)
        a = randComp(-100, 100);
    vector<Comp> B(N), C(N);
    naive.dft(B.data(), A.data());
    naive.idft(C.data(), A.data());
    [&](auto...){}([&](auto&& fft){
        vector<Comp> D(N), E(N);
        fft.dft(D.data(), A.data());
        fft.idft(E.data(), A.data());
        auto err = error(B.data(), D.data(), N);
        cout << "[" << fft.Name << "] err(dft(x), fft(x)) = " << err << endl;
        err = error(C.data(), E.data(), N);
        cout << "[" << fft.Name << "] err(idft(x), ifft(x)) = " << err << endl;
        return 0;
    }(ffts)...);
}

void testDifDit(const DifDitFFT& T) {
    using namespace std;
    auto N = T.N;
    auto M = N / 2;
    vector<Comp> A(N), B(N), C(N), D(N);
    for (size_t i = 0; i < M; ++i) {
        A[i] = randComp(-100, 100);
        B[i] = randComp(-100, 100);
    }
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < M; ++j)
            C[i + j] += A[i] * B[j];
    T.dft(A.data());
    T.dft(B.data());
    for (size_t i = 0; i < N; ++i)
        D[i] = A[i] * B[i];
    T.idft(D.data());
    auto err = error(C.data(), D.data(), M * 2 - 1);
    cout << "[" << T.Name << "] err(a*b, idft(dft(a)*dft(b))) = " << err << endl;
}

void testDifDit_cuda(DifDitFFT_cuda& T) {
    using namespace std;
    auto N = T.N;
    auto M = N / 2;
    vector<Comp> A(N), B(N);
    for (size_t i = 0; i < M; ++i) {
        A[i] = randComp(-100, 100);\
        B[i] = Comp(A[i].real(), A[i].imag());
    }
    T.dft(A.data());
    T.idft(A.data());
    auto err = error(A.data(), B.data(), M * 2 - 1);
    cout << "[" << T.Name << "] err(x, idft(dft(x))) = " << err << endl;

    // vector<Comp> C(N), D(N);
    // for (size_t i = 0; i < M; ++i)
    //     B[i] = randComp(-100, 100);

    // dim3 blockDim(32, 32);
    // int col_s = (N + blockDim.y - 1) / blockDim.y;
    // dim3 gridDim((N + blockDim.x - 1) / blockDim.x, col_s);

    // cuDoubleComplex* h_A = (cuDoubleComplex*) malloc(N*sizeof(cuDoubleComplex));
    // cuDoubleComplex* h_B = (cuDoubleComplex*) malloc(N*sizeof(cuDoubleComplex));
    // cuDoubleComplex* h_C = (cuDoubleComplex*) malloc(N*col_s*sizeof(cuDoubleComplex));
    // cuDoubleComplex* h_D = (cuDoubleComplex*) malloc(N*sizeof(cuDoubleComplex));

    // Comp_to_cuComp(A.data(), h_A, N);
    // Comp_to_cuComp(B.data(), h_B, N);

    // cuDoubleComplex* d_A;
    // cuDoubleComplex* d_B;
    // cuDoubleComplex* d_C; 
    // // cuDoubleComplex* d_D;

    // cudaMalloc(&d_A, N*sizeof(cuDoubleComplex));
    // cudaMalloc(&d_B, N*sizeof(cuDoubleComplex));
    
    // cudaMalloc(&d_C, N*col_s*sizeof(cuDoubleComplex));
    // // cudaMalloc(&d_D, N*sizeof(cuDoubleComplex));

    // cudaMemcpy(d_A, A.data(), N*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_B, B.data(), N*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    
    // complex_vec_mul_kernel<<<(N + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE>>>(d_A, d_B, d_C, N, col_s);

    // reduction_2d<<<(N + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE>>>(d_C, N, col_s);

    // cudaMemcpy(h_C, d_C, N*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    // for(int i = 0; i < N; ++i){
    //     C[i] = Comp(cuCreal(h_C[i]), cuCimag(h_C[i]));
    // }

    // T.dft(A.data());
    // T.dft(B.data());

    // for (size_t i = 0; i < N; ++i)
    //     D[i] = A[i] * B[i];

    // T.idft(D.data());
    // err = error(C.data(), D.data(), M * 2 - 1);
    // cout << "[" << T.Name << "] err(a*b, idft(dft(a)*dft(b))) = " << err << endl;

    // free(h_A);
    // free(h_B);
    // free(h_C);
    // cudaFree(d_A);
    // cudaFree(d_B);
    // cudaFree(d_C);
}

void testBasic() {
    constexpr size_t N = 1024*1024;
    // NaiveDFT naive(N);
    // NaiveDFT_cuda naive_cuda(N);
    DitFFT dit(N);
    DifDitFFT difdit(N);
    DitFFT_cuda dit_cuda(N);
    DifDitFFT_cuda difdit_cuda(N);
    // DifFFT dif(N);
    // testAll(naive);
    // testAll(naive_cuda);
    testAll(dit);
    testAll(dit_cuda);
    // testAll(dif);
    // testCmp(naive, dit, dif, naive_cuda);
    // testDifDit(difdit);
    testDifDit_cuda(difdit_cuda);
}

void testArbitrary() {
    using namespace std;
    constexpr size_t N = 1000;
    NaiveDFT naive(N);
    NaiveDFT_cuda naive_cuda(N);
    BluesteinFFT bluestein(N);
    testAll(bluestein);
    testAll(naive_cuda);
    testCmp(naive, bluestein);
}

int main() {
    using namespace std;
    testBasic();
  //  testArbitrary();
    return 0;
}
