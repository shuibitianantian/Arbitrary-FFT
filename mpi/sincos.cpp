#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>

using namespace std;
using namespace chrono;

using Real = double;
using Comp = complex<Real>;

constexpr ptrdiff_t Alignment = 64;

template<class T>
T* alloc(ptrdiff_t N) {
  if (auto res = aligned_alloc(Alignment, N * sizeof(T)))
    return (T*) res;
  fprintf(stderr, "Allocation of %zu bytes failed\n", N * sizeof(T));
  exit(EXIT_FAILURE);
}

constexpr Real Pi = 3.1415926535897932384626433832795;
constexpr Real _2Pi = 6.283185307179586476925286766559;

int main() {
  constexpr int K = 100;
  constexpr ptrdiff_t M = (ptrdiff_t) 1 << 20;
  auto A = alloc<Comp>(M);
  auto B = alloc<Comp>(M);
  auto w = _2Pi / (Real) M;
  for (volatile int k = 0; k < K; ++k)
    for (ptrdiff_t i = 0; i < M; ++i)
      B[i] = {cos(w * (Real) i), sin(w * (Real) i)};
  auto t3 = high_resolution_clock::now();
  for (volatile int k = 0; k < K; ++k)
    for (ptrdiff_t i = 0; i < M; ++i)
      B[i] = {cos(w * (Real) i), sin(w * (Real) i)};
  auto t4 = high_resolution_clock::now();
  for (volatile int k = 0; k < K; ++k)
    for (ptrdiff_t i = 0; i < M; ++i)
      A[i] = exp(Comp(0, w * (Real) i));
  auto t1 = high_resolution_clock::now();
  for (volatile int k = 0; k < K; ++k)
    for (ptrdiff_t i = 0; i < M; ++i)
      A[i] = exp(Comp(0, w * (Real) i));
  auto t2 = high_resolution_clock::now();
  auto ta = duration_cast<nanoseconds>(t2 - t1).count() * 1e-9;
  auto tb = duration_cast<nanoseconds>(t4 - t3).count() * 1e-9;
  Real err = 0;
  for (ptrdiff_t i = 0; i < M; ++i)
    err += abs(A[i] - B[i]);
  printf("err: %g\n", err);
  printf("exp: %f s\n", ta);
  printf("cis: %f s\n", tb);
  free(A);
  free(B);
  return 0;
}
