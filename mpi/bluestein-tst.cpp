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
    W = alloc<Comp>(L);
    B = alloc<Comp>(M);
    C = alloc<Comp>(N);

    for (int i = 0; i < L; ++i)
      W[i] = cis(ldexp(_2Pi, -i - 1));

    auto w = Pi / (Real) N;
    if (!sign)
      w = -w;
    for (ptrdiff_t i = 0; i < N; ++i)
      C[i] = cis(w * (Real) (i * i));

    B[0] = conj(C[0]);
    for (ptrdiff_t i = 1; i < N; ++i)
      B[i] = B[M - i] = conj(C[i]);
    fill(B + N, B + M - N + 1, Comp{});

    dft(B);

    ofstream fb("bb.dat", ios::binary);
    fb.write((char*) B, M * sizeof(Comp));

    ofstream fc("bc.dat", ios::binary);
    fc.write((char*) C, N * sizeof(Comp));
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
      auto wn = W[i];
      auto h = ptrdiff_t{1} << i;
      for (ptrdiff_t j = 0; j < M; j += h << 1) {
        Comp w = (Real) 1;
        for (ptrdiff_t k = 0; k < h; ++k) {
          auto u = Y[j + k];
          auto v = Y[j + k + h];
          Y[j + k] = u + v;
          Y[j + k + h] = w * (u - v);
          w *= wn;
        }
      }
    }
  }

  void idft(Comp* Y) const {
    for (int i = 0; i < L; ++i) {
      auto wn = conj(W[i]);
      auto h = ptrdiff_t{1} << i;
      for (ptrdiff_t j = 0; j < M; j += h << 1) {
        Comp w = (Real) 1;
        for (ptrdiff_t k = 0; k < h; ++k) {
          auto u = Y[j + k];
          auto v = w * Y[j + k + h];
          Y[j + k] = u + v;
          Y[j + k + h] = u - v;;
          w *= wn;
        }
      }
    }
    for (ptrdiff_t i = 0; i < M; ++i)
      //Y[i] = Y[i].ldexp(-L);
      Y[i] /= (Real) M;
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
