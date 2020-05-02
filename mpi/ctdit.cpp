#include <algorithm>
#include <chrono>
#include <climits>
#include <complex>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <utility>

using namespace std;
using namespace chrono;

using Real = double;
using Comp = complex<Real>;

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

constexpr Real _2Pi = 6.283185307179586476925286766559;

// Decimation in Time FFT
// Performs bit-reverse-copy before transform
// N must be power of 2
struct CtDit {
  const int L;
  const ptrdiff_t N;
  ptrdiff_t NI;
  Comp* W;
  ptrdiff_t (*I)[2];

  CtDit(ptrdiff_t N, bool sign) : L(__builtin_ctzll(N)), N(N), NI(0) {
    using namespace std;
    W = alloc<Comp>(N);
    I = alloc<ptrdiff_t[2]>(N);
    auto w = _2Pi / (Real) N;
    if (!sign)
      w = -w;
    for (ptrdiff_t i = 0; i < N; ++i)
      W[i] = exp(Comp(0, w * (Real) i));
    for (ptrdiff_t i = 0, j = 0; i < N; ++i) {
      if (i < j) {
        I[NI][0] = i;
        I[NI][1] = j;
        ++NI;
      }
      for (ptrdiff_t k = N >> 1; (j ^= k) < k; k >>= 1);
    }
  }

  ~CtDit() {
    free(W);
    free(I);
  }

  void transform(Comp* Y) const {
    for (ptrdiff_t i = 0; i < NI; ++i)
      swap(Y[I[i][0]], Y[I[i][1]]);
    for (int i = 0; i < L; ++i) {
      auto g = N >> (i + 1);
      auto h = ptrdiff_t{1} << i;
      for (ptrdiff_t j = 0; j < N; j += h << 1)
        for (ptrdiff_t k = 0; k < h; ++k) {
          auto u = Y[j + k];
          auto v = W[k * g] * Y[j + k + h];
          Y[j + k] = u + v;
          Y[j + k + h] = u - v;;
        }
    }
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

  CtDit ctdit(N, sign);

  auto Y = alloc<Comp>(N);

  if (!fi.read((char*) Y, sizeof(Comp) * N)) {
    fprintf(stderr, "Failed to load %zd complex numbers\n", N);
    return EXIT_FAILURE;
  }

  printf("FFT starts\n");
  auto tStart = high_resolution_clock::now();

  ctdit.transform(Y);
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

  if (!fo.write((char*) Y, sizeof(Comp) * N)) {
    fprintf(stderr, "Failed to save %zd complex numbers\n", N);
    return EXIT_FAILURE;
  }

  free(Y);

  printf("All done\n");

  return 0;
}
