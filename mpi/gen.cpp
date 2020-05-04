#include <complex>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>

using namespace std;

using Real = double;
using Comp = complex<Real>;

bool checkArgs(ptrdiff_t& N, char* fname, int nArg, char* args[]) {
  if (nArg < 2 || nArg > 3) {
    fprintf(stderr, "Incorrect command line\n");
    return false;
  }
  if (1 != sscanf(args[1], "%zd", &N) || N < 1) {
    fprintf(stderr, "Invalid N: %s\n", args[1]);
    return false;
  }
  if (nArg >= 3)
    strcpy(fname, args[2]);
  else {
    sprintf(fname, "n%zd.dat", N);
  }
  return true;
}

int main(int nArg, char* args[]) {
  ptrdiff_t N;
  static char fname[256];
  if (!checkArgs(N, fname, nArg, args)) {
    fprintf(stderr, "Usage: %s <N> [Output]\n", args[0]);
    return EXIT_FAILURE;
  }

  ofstream fo(fname, ios::binary);
  if (!fo) {
    fprintf(stderr, "Failed to open file for writing: %s\n", fname);
    return EXIT_FAILURE;
  }

  mt19937_64 rand(random_device{}());

  Real val[2];
  for (ptrdiff_t i = 0; i < N; ++i) {
    val[0] = uniform_real_distribution<Real>(-100, 100)(rand);
    val[1] = uniform_real_distribution<Real>(-100, 100)(rand);
    if (!fo.write((char*) val, sizeof(val))) {
      fprintf(stderr, "Failed to write %zu bytes\n", sizeof(val));
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}
