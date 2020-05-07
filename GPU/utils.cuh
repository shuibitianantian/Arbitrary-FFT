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