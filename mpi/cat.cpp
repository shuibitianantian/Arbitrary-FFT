#include <complex>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

using namespace std;

using Real = double;
using Comp = complex<Real>;

bool checkArgs(char*& fname, ptrdiff_t& off, ptrdiff_t& n,
  int nArg, char* args[])
{
  if (nArg < 3 || nArg > 4) {
    fprintf(stderr, "Incorrect command line\n");
    return false;
  }
  fname = args[1];
  if (1 != sscanf(args[2], "%zd", &n) || n < 0) {
    fprintf(stderr, "Invalid Count: %s\n", args[2]);
    return false;
  }
  off = 0;
  if (nArg > 3 && (1 != sscanf(args[3], "%zd", &off) || off < 0)) {
    fprintf(stderr, "Invalid Offset: %s\n", args[3]);
    return false;
  }
  return true;
}

int main(int nArg, char* args[]) {
  ptrdiff_t off, n;
  char* fname;
  if (!checkArgs(fname, off, n, nArg, args)) {
    fprintf(stderr, "Usage: %s <File> <Count> [Offset]\n", args[0]);
    return EXIT_FAILURE;
  }

  ifstream fi(fname, ios::binary);

  if (!fi) {
    fprintf(stderr, "Failed to open file for reading: %s\n", fname);
    return EXIT_FAILURE;
  }

  fi.seekg(off * sizeof(Comp));
  for (ptrdiff_t i = 0; i < n; ++i) {
    Comp c;
    if (!fi.read((char*) &c, sizeof(Comp))) {
      fprintf(stderr, "Failed to read %zu bytes from %s\n", sizeof(c), fname);
      return EXIT_FAILURE;
    }
    printf("%g + i * %g\n", c.real(), c.imag());
  }

  return EXIT_SUCCESS;
}
