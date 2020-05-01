#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>
#include <cuComplex.h>
#include <stdio.h>
#include <omp.h>
#include <cufft.h>          //CUFFT文件头

#ifdef _MSC_VER
#include <intrin.h>
typedef unsigned long DWORD;
uint32_t __inline __builtin_ctzll(uint32_t value )
{
    DWORD trailing_zero = 0;

    if ( _BitScanForward( &trailing_zero, value ) )
    {
        return trailing_zero;
    }
    else
    {
        // This is undefined, I better choose 32 than 0
        return 32;
    }
}

#endif

__device__ double atomicAdd2(double* address, double val)
{
    unsigned long long int* address_as_ull =
            (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

__device__ cuDoubleComplex cuda_complex_exp (cuDoubleComplex arg){
    double s, c;
    double e = exp(cuCreal(arg));
    sincos(cuCimag(arg), &s, &c);
    return make_cuDoubleComplex(c * e, s * e);
}

void Check_CUDA_Error(const char *message){
    cudaError_t error = cudaGetLastError();
    if(error!=cudaSuccess) {
        fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
        exit(-1);
    }
}

using Real = double;
using Comp = ::std::complex<Real>;

constexpr Real Pi = 3.1415926535897932384626433832795;
constexpr Real _2Pi = 6.283185307179586476925286766559;

// Naive implementation of DFT
// Arbitrary N
#define BLOCKSIZE 512
__device__ double img;
__device__ double rl;

//cuDoubleComplex c = make_cuDoubleComplex(cr, ci);
//cuDoubleComplex result = cuCmul(c, make_cuDoubleComplex(r, 0));

__global__ void dft_kernel(cuDoubleComplex* X, const std::size_t N, Real k){
    __shared__ cuDoubleComplex cache[BLOCKSIZE];
    std::size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N){
        cuDoubleComplex tmp = cuda_complex_exp(make_cuDoubleComplex(0, -_2Pi / (Real) N * (Real) k * (Real) id));
        cache[threadIdx.x] = cuCmul(tmp, X[id]);
        __syncthreads();
        // perform reduction in block
        for (unsigned int s = blockDim.x/2; s>0; s>>=1) {
            if (threadIdx.x < s) {
                cache[threadIdx.x] = cuCadd(cache[threadIdx.x], cache[threadIdx.x + s]);
            }
            __syncthreads();
        }
    }

    if(threadIdx.x == 0){
        atomicAdd2(&img, cuCimag(cache[threadIdx.x]));
        atomicAdd2(&rl, cuCreal(cache[threadIdx.x]));
    }

}

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
        cout << omp_get_wtime() - tt << endl;
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

    void cuComp_to_Comp(cuDoubleComplex* cucp, Comp* cp, std::size_t N){
        for(int i = 0; i < N; ++i){
            cp[i] = Comp(cuCreal(cucp[i]), cuCimag(cucp[i]));
        }
    }

    void Comp_to_cuComp(const Comp* cp, cuDoubleComplex* cucp, std::size_t N){
        for(int i = 0; i < N; ++i){
            cucp[i] = make_cuDoubleComplex(cp[i].real(), cp[i].imag());
        }
    }

    void dft(Comp* Y, const Comp* X){
        using namespace std;

        cuDoubleComplex* h_X = (cuDoubleComplex*) malloc(N*sizeof(cuDoubleComplex));
        Comp_to_cuComp(X, h_X, N);  // convert Comp to cuda complex
        cuDoubleComplex* d_X;

        cudaMalloc(&d_X, N*sizeof(cuDoubleComplex));
        cudaMemcpy(d_X, h_X, N*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

        Real h_real = 0.0;
        Real h_img = 0.0;

        cudaMemcpyToSymbol(rl, &h_real, sizeof(Real));
        cudaMemcpyToSymbol(img, &h_img, sizeof(Real));
        cudaDeviceSynchronize();

        double tt = omp_get_wtime();
        for(int i = 0; i < N; ++i) {
            dft_kernel<<<(N + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE>>>(d_X, N, i);
            cudaMemcpyFromSymbol(&h_real, rl, sizeof(Real));
            cudaMemcpyFromSymbol(&h_img, img, sizeof(Real));
            cudaDeviceSynchronize();

            Y[i] = Comp(h_real, h_img);

            h_real = h_img = 0.0;
            cudaMemcpyToSymbol(rl, &h_real, sizeof(Real));
            cudaMemcpyToSymbol(img, &h_img, sizeof(Real));
            cudaDeviceSynchronize();
        }

        cout << omp_get_wtime() - tt << endl;
        free(h_X);
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
    }

    void idft(Comp* Y, const Comp* X) const {
        using namespace std;
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
    }

    void idft(Comp* Y) const {
        using namespace std;
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
    testConvolution(std::forward<Tr>(T));
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

void testBasic() {
    constexpr size_t N = 1024*4;
    NaiveDFT naive(N);
    NaiveDFT_cuda naive_cuda(N);
//    NaiveDFT_cudafft naive_cudafft(N);
    DitFFT dit(N);
    DifFFT dif(N);
    DifDitFFT difdit(N);
    testAll(naive);
    testAll(naive_cuda);
//    testAll(naive_cudafft);
    testAll(dit);
    testAll(dif);
    testCmp(naive, dit, dif, naive_cuda);
    testDifDit(difdit);
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
//    testArbitrary();
    return 0;
}
