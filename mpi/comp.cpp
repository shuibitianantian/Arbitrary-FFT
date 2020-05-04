#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>

using namespace std;
using namespace chrono;

#include "common.h"

using Cstd = complex<Real>;

constexpr ptrdiff_t Alignment = 64;

template<class T>
T* alloc(ptrdiff_t N) {
  if (auto res = aligned_alloc(Alignment, N * sizeof(T)))
    return (T*) res;
  fprintf(stderr, "Allocation of %zu bytes failed\n", N * sizeof(T));
  exit(EXIT_FAILURE);
}

int main() {
  constexpr int K = 100;
  constexpr int L = 11;
  constexpr ptrdiff_t M = (ptrdiff_t) 1 << L;
  auto A = alloc<Cstd>(M);
  auto B = alloc<Comp>(M);
  auto C = alloc<Cstd>(M);
  auto D = alloc<Comp>(M);
  auto wn = _2Pi / (Real) M;
  for (ptrdiff_t i = 0; i < M; ++i) {
    auto w = wn * (Real) i;
    auto c = cos(w);
    auto s = sin(w);
    A[i] = {c, s};
    B[i] = {c, s};
  }
  for (volatile int k = 0; k < K; ++k)
    for (ptrdiff_t i = 0; i < M; ++i) {
      Cstd v{1, 0};
      for (ptrdiff_t j = 0; j < M; ++j)
        if (i != j)
          v *= A[j];
      C[i] = v;
    }
  auto t1 = high_resolution_clock::now();
  for (volatile int k = 0; k < K; ++k)
    for (ptrdiff_t i = 0; i < M; ++i) {
      Cstd v{1, 0};
      for (ptrdiff_t j = 0; j < M; ++j)
        if (i != j)
          v *= A[j];
      C[i] = v;
    }
  auto t2 = high_resolution_clock::now();
  for (volatile int k = 0; k < K; ++k)
    for (ptrdiff_t i = 0; i < M; ++i) {
      Comp v{1, 0};
      for (ptrdiff_t j = 0; j < M; ++j)
        if (i != j)
          v *= B[j];
      D[i] = v;
    }
  auto t3 = high_resolution_clock::now();
  for (volatile int k = 0; k < K; ++k)
    for (ptrdiff_t i = 0; i < M; ++i) {
      Comp v{1, 0};
      for (ptrdiff_t j = 0; j < M; ++j)
        if (i != j)
          v *= B[j];
      D[i] = v;
    }
  auto t4 = high_resolution_clock::now();
  auto ta = duration_cast<nanoseconds>(t2 - t1).count() * 1e-9;
  auto tb = duration_cast<nanoseconds>(t4 - t3).count() * 1e-9;
  Real err = 0;
  for (ptrdiff_t i = 0; i < L; ++i)
    err += hypot(A[i].real() - B[i].re, A[i].imag() - B[i].im);
  printf("err: %g\n", err);
  printf("cstd: %f s\n", ta);
  printf("comp: %f s\n", tb);
  free(A);
  free(B);
  free(C);
  free(D);
  return 0;
}
