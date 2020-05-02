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

constexpr size_t Alignment = 64;

template<class T>
T* alloc(size_t N) {
  if (auto res = std::aligned_alloc(Alignment, N * sizeof(T)))
    return (T*) res;
  fprintf(stderr, "Allocation of %zu bytes failed\n", N * sizeof(T));
  exit(EXIT_FAILURE);
}

constexpr Real _2Pi = 6.283185307179586476925286766559;

// Naive implementation of DFT
// Arbitrary N
struct NaiveDFT {
  const ptrdiff_t N;

  NaiveDFT(ptrdiff_t N) : N(N) {}

  void dft(Comp* Y, const Comp* X) const {
    for (ptrdiff_t k = 0; k < N; ++k) {
      Comp y = 0;
      for (ptrdiff_t n = 0; n < N; ++n)
        y += exp(Comp(0, -_2Pi / (Real) N * (Real) k * (Real) n)) * X[n];
      Y[k] = y;
    }
  }

  void idft(Comp* Y, const Comp* X) const {
    for (ptrdiff_t k = 0; k < N; ++k) {
      Comp y = 0;
      for (ptrdiff_t n = 0; n < N; ++n)
        y += exp(Comp(0, _2Pi / (Real) N * (Real) k * (Real) n)) * X[n];
      Y[k] = y / (Real) N;
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

  NaiveDFT naive(N);

  auto X = alloc<Comp>(N);
  auto Y = alloc<Comp>(N);

  if (!fi.read((char*) X, sizeof(Comp) * N)) {
    fprintf(stderr, "Failed to load %zd complex numbers\n", N);
    return EXIT_FAILURE;
  }

  printf("DFT starts\n");
  auto tStart = high_resolution_clock::now();

  if (sign)
    naive.idft(Y, X);
  else
    naive.dft(Y, X);

  auto tEnd = high_resolution_clock::now();
  auto time = duration_cast<nanoseconds>(tEnd - tStart).count() * 1e-9;
  printf("DFT ends\n");
  printf("Time: %f s\n", time);

  if (!fo.write((char*) Y, sizeof(Comp) * N)) {
    fprintf(stderr, "Failed to save %zd complex numbers\n", N);
    return EXIT_FAILURE;
  }

  free(X);
  free(Y);

  printf("All done\n");

  return 0;
}
