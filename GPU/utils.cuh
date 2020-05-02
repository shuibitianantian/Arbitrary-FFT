// ============= Global configurations ==============
#include <cuComplex.h>
#include <complex>

#define BLOCKSIZE 1024
using Real = double;
using Comp = ::std::complex<Real>;

constexpr Real Pi = 3.1415926535897932384626433832795;
constexpr Real _2Pi = 6.283185307179586476925286766559;

// =============== Windows version ctz =================
#ifdef _MSC_VER
#include <intrin.h>
typedef unsigned long DWORD;
uint32_t __inline __builtin_ctzll(uint32_t value ){
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
// =====================================================

// Self defined atomicAdd
__device__ double atomicAdd2(double* address, double val){
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