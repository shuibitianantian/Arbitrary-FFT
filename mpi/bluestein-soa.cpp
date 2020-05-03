#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <utility>

using namespace std;
using namespace chrono;

#include "common.h"

bool checkArgs(bool& sign, ptrdiff_t& N, char*& fnamei, char*& fnameo,
  int nArg, char* args[])
{
  if (nArg != 5) {
    fprintf(stderr, "Incorrect command line\n");
    return false;
  }
  int b;
  if (1 != sscanf(args[1], "%d", &b) || (b & ~1)) {
    fprintf(stderr, "Invalid sign: %s\n", args[1]);
    return false;
  }
  sign = b;
  if (1 != sscanf(args[2], "%zd", &N) || N < 1) {
    fprintf(stderr, "Invalid N: %s\n", args[1]);
    return false;
  }
  fnamei = args[3];
  fnameo = args[4];
  return true;
}

constexpr ptrdiff_t Alignment = 64;

template<class T>
T* alloc(ptrdiff_t N) {
  if (auto res = aligned_alloc(Alignment, N * sizeof(T)))
    return (T*) res;
  fprintf(stderr, "Allocation of %zu bytes failed\n", N * sizeof(T));
  exit(EXIT_FAILURE);
}

struct Comps {
  Real* re = nullptr;
  Real* im = nullptr;

  constexpr Comps() noexcept = default;
  inline Comps(size_t N) { alloc(N); }

  void alloc(size_t N) {
    re = ::alloc<Real>(N);
    im = ::alloc<Real>(N);
  }

  ~Comps() {
    free(re);
    free(im);
  }

  struct Proxy {
    Real &re, &im;
    constexpr operator Comp() const noexcept { return {re, im}; }
    constexpr Proxy& operator =(const Comp& z) noexcept {
      re = z.re;
      im = z.im;
      return *this;
    }
  };

  constexpr void set(ptrdiff_t i, const Comp& z) {
    re[i] = z.re;
    im[i] = z.im;
  }

  constexpr Comp operator [](ptrdiff_t i) const {
    return {re[i], im[i]};
  }

  //constexpr Proxy operator [](ptrdiff_t i) {
  //  return {re[i], im[i]};
  //}
};

// Bluestein's Algorithm: Chirp Z-Transform
// Arbitrary N
struct Bluestein {
  const ptrdiff_t N;
  const ptrdiff_t M;
  const int L;
  Comp *W;
  Comps B, C;

  Bluestein(ptrdiff_t N, bool sign) : N(N), M(ceil2(N + N - 1)),
    L(__builtin_ctzll(M)), B(M), C(N)
  {
    W = alloc<Comp>(L);

    for (int i = 0; i < L; ++i)
      W[i] = cis(ldexp(_2Pi, -i - 1));

    auto w = Pi / (Real) N;
    if (!sign)
      w = -w;
    for (ptrdiff_t i = 0; i < N; ++i)
      C[i] = cis(w * (Real) (i * i));

    B[0] = conj(C[0]);
    for (ptrdiff_t i = 1; i < N; ++i) {
      B[i] = conj(C[i]);
      B[M - i] = conj(C[i]);
    }
    fill(B.re + N, B.re + M - N + 1, Real{});
    fill(B.im + N, B.im + M - N + 1, Real{});
    dft(B);
  }

  ~Bluestein() {
    free(W);
  }

  void transform(Comps& Y) {
    if (M != ((ptrdiff_t) 1 << L))
      __builtin_unreachable();
    for (ptrdiff_t i = 0; i < N; ++i)
      Y.set(i, Y[i] * C[i]);
    fill(Y.re + N, Y.re + M, Real{});
    fill(Y.im + N, Y.im + M, Real{});
    dft(Y);
    for (ptrdiff_t i = 0; i < M; ++i)
      Y[i] *= B[i];
    idft(Y);
    for (ptrdiff_t i = 0; i < N; ++i)
      Y[i] *= C[i];
  }

  void dft(Comps& Y) const {
    if (M != ((ptrdiff_t) 1 << L))
      __builtin_unreachable();
    for (int i = L - 1; i >= 0; --i) {
      auto wn = W[i];
      auto h = ptrdiff_t{1} << i;
      for (ptrdiff_t j = 0; j < M; j += h << 1) {
        Comp w = (Real) 1;
        for (ptrdiff_t k = 0; k < h; ++k) {
          auto u = Y[j + k];
          auto v = Y[j + k + h];
          Y.set(j + k, u + v);
          Y.set(j + k + h, w * (u - v));
          w *= wn;
        }
      }
    }
  }

  void idft(Comps& Y) const {
    if (M != ((ptrdiff_t) 1 << L))
      __builtin_unreachable();
    for (int i = 0; i < L; ++i) {
      auto wn = conj(W[i]);
      auto h = ptrdiff_t{1} << i;
      for (ptrdiff_t j = 0; j < M; j += h << 1) {
        Comp w = (Real) 1;
        for (ptrdiff_t k = 0; k < h; ++k) {
          auto u = Y[j + k];
          auto v = w * Y[j + k + h];
          Y.set(j + k, u + v);
          Y.set(j + k + h, u - v);
          w *= wn;
        }
      }
    }
    for (ptrdiff_t i = 0; i < M; ++i)
      Y.re[i] /= (Real) M;
    for (ptrdiff_t i = 0; i < M; ++i)
      Y.im[i] /= (Real) M;
      //Y[i] = Y[i].ldexp(-L);
      //Y[i] /= (Real) M;
  }
};

int main(int nArg, char* args[]) {
  bool sign;
  ptrdiff_t N;
  char *fnamei, *fnameo;
  if (!checkArgs(sign, N, fnamei, fnameo, nArg, args)) {
    fprintf(stderr, "Usage: %s <sign> <N> <Input> <Output>\n", args[0]);
    return EXIT_FAILURE;
  }

  ifstream fi(fnamei, ios::binary);
  ofstream fo(fnameo, ios::binary);

  if (!fi) {
    fprintf(stderr, "Failed to open file for reading: %s\n", fnamei);
    return EXIT_FAILURE;
  }

  if (!fo) {
    fprintf(stderr, "Failed to open file for writing: %s\n", fnameo);
    return EXIT_FAILURE;
  }

  Bluestein bluestein(N, sign);

  Comps Y(bluestein.M);

  for (ptrdiff_t i = 0; i < N; ++i) {
    Comp z;
    if (!fi.read((char*) &z, sizeof(Comp))) {
      fprintf(stderr, "Failed to load %zd complex numbers\n", N);
      return EXIT_FAILURE;
    }
    Y.set(i, z);
  }

  printf("FFT starts\n");
  auto tStart = high_resolution_clock::now();

  bluestein.transform(Y);
  if (sign) {
    // Does not divide the result by N
    auto inv = 1 / (Real) N;
    for (ptrdiff_t i = 0; i < N; ++i)
      Y[i] *= inv;
  }

  auto tEnd = high_resolution_clock::now();
  auto time = duration_cast<nanoseconds>(tEnd - tStart).count() * 1e-9;
  printf("FFT ends\n");
  printf("Time: %f s\n", time);

  for (ptrdiff_t i = 0; i < N; ++i) {
    Comp z = Y[i];
    if (!fo.write((char*) &z, sizeof(Comp))) {
      fprintf(stderr, "Failed to save %zd complex numbers\n", N);
      return EXIT_FAILURE;
    }
    Y.set(i, z);
  }

  printf("All done\n");

  return 0;
}
