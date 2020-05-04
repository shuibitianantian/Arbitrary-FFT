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
  constexpr int K = 10000000;
  constexpr int L = 20;
  constexpr ptrdiff_t M = (ptrdiff_t) 1 << L;
  auto A = alloc<Comp>(L);
  auto B = alloc<Comp>(L);
  auto wn = _2Pi / (Real) M;
  for (volatile int k = 0; k < K; ++k)
    for (int i = 0; i < L; ++i) {
      auto w = ldexp(_2Pi, i - L);
      A[i] = {cos(w), sin(w)};
    }
  auto t1 = high_resolution_clock::now();
  for (volatile int k = 0; k < K; ++k)
    for (int i = 0; i < L; ++i) {
      auto w = ldexp(_2Pi, i - L);
      A[i] = {cos(w), sin(w)};
    }
  auto t2 = high_resolution_clock::now();
  for (volatile int k = 0; k < K; ++k)
    for (int i = 0; i < L; ++i) {
      auto w = wn * (Real) ((ptrdiff_t) 1 << i);
      B[i] = {cos(w), sin(w)};
    }
  auto t3 = high_resolution_clock::now();
  for (volatile int k = 0; k < K; ++k)
    for (int i = 0; i < L; ++i) {
      auto w = wn * (Real) ((ptrdiff_t) 1 << i);
      B[i] = {cos(w), sin(w)};
    }
  auto t4 = high_resolution_clock::now();
  auto ta = duration_cast<nanoseconds>(t2 - t1).count() * 1e-9;
  auto tb = duration_cast<nanoseconds>(t4 - t3).count() * 1e-9;
  Real err = 0;
  for (ptrdiff_t i = 0; i < L; ++i)
    err += abs(A[i] - B[i]);
  printf("err: %g\n", err);
  printf("divmul: %f s\n", ta);
  printf("ldexp: %f s\n", tb);
  free(A);
  free(B);
  return 0;
}
