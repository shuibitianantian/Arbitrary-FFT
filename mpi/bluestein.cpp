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

constexpr Real Pi = 3.1415926535897932384626433832795;
constexpr Real _2Pi = 6.283185307179586476925286766559;

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

// Bluestein's Algorithm: Chirp Z-Transform
// Arbitrary N
struct Bluestein {
  const ptrdiff_t N;
  const ptrdiff_t M;
  const int L;
  Comp *W, *B, *C;

  Bluestein(ptrdiff_t N, bool sign) : N(N), M(ceil2(N + N - 1)),
    L(__builtin_ctzll(M))
  {
    W = alloc<Comp>(M);
    B = alloc<Comp>(M);
    C = alloc<Comp>(N);

    auto w = _2Pi / (Real) M;
    for (ptrdiff_t i = 0; i < M; ++i)
      W[i] = exp(Comp(0, w * (Real) i));

    w = Pi / (Real) N;
    if (!sign)
      w = -w;
    for (ptrdiff_t i = 0; i < N; ++i)
      C[i] = exp(Comp(0, w * (Real) (i * i)));

    B[0] = conj(C[0]);
    for (ptrdiff_t i = 1; i < N; ++i)
      B[i] = B[M - i] = conj(C[i]);
    dft(B);
  }

  ~Bluestein() {
    free(W);
    free(B);
    free(C);
  }

  void transform(Comp* Y) {
    for (ptrdiff_t i = 0; i < N; ++i)
      Y[i] *= C[i];
    fill(Y + N, Y + M, Comp{});
    dft(Y);
    for (ptrdiff_t i = 0; i < M; ++i)
      Y[i] *= B[i];
    idft(Y);
    for (ptrdiff_t i = 0; i < N; ++i)
      Y[i] *= C[i];
  }

  void dft(Comp* Y) const {
    for (int i = L - 1; i >= 0; --i) {
      auto g = M >> (i + 1);
      auto h = ptrdiff_t{1} << i;
      for (ptrdiff_t j = 0; j < M; j += h << 1)
        for (ptrdiff_t k = 0; k < h; ++k) {
          auto u = Y[j + k];
          auto v = Y[j + k + h];
          Y[j + k] = u + v;
          Y[j + k + h] = conj(W[k * g]) * (u - v);
        }
    }
  }

  void idft(Comp* Y) const {
    for (int i = 0; i < L; ++i) {
      auto g = M >> (i + 1);
      auto h = ptrdiff_t{1} << i;
      for (ptrdiff_t j = 0; j < M; j += h << 1)
        for (ptrdiff_t k = 0; k < h; ++k) {
          auto u = Y[j + k];
          auto v = W[k * g] * Y[j + k + h];
          Y[j + k] = u + v;
          Y[j + k + h] = u - v;;
        }
    }
    auto inv = 1 / (Real) M;
    for (ptrdiff_t i = 0; i < M; ++i)
      Y[i] *= inv;
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

  auto Y = alloc<Comp>(bluestein.M);

  if (!fi.read((char*) Y, sizeof(Comp) * N)) {
    fprintf(stderr, "Failed to load %zd complex numbers\n", N);
    return EXIT_FAILURE;
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

  if (!fo.write((char*) Y, sizeof(Comp) * N)) {
    fprintf(stderr, "Failed to save %zd complex numbers\n", N);
    return EXIT_FAILURE;
  }

  free(Y);

  printf("All done\n");

  return 0;
}
