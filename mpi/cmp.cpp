#include <complex>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

using namespace std;

using Real = double;
using Comp = complex<Real>;

bool checkArgs(ptrdiff_t& N, char*& fname1, char*& fname2,
  int nArg, char* args[])
{
  if (nArg != 4) {
    fprintf(stderr, "Incorrect command line\n");
    return false;
  }
  if (1 != sscanf(args[1], "%zd", &N) || N < 1) {
    fprintf(stderr, "Invalid N: %s\n", args[1]);
    return false;
  }
  fname1 = args[2];
  fname2 = args[3];
  return true;
}

int main(int nArg, char* args[]) {
  ptrdiff_t N;
  char *fname1, *fname2;
  if (!checkArgs(N, fname1, fname2, nArg, args)) {
    fprintf(stderr, "Usage: %s <N> <File1> <File2>\n", args[0]);
    return EXIT_FAILURE;
  }

  ifstream f1(fname1, ios::binary);
  ifstream f2(fname2, ios::binary);

  if (!f1) {
    fprintf(stderr, "Failed to open file for reading: %s\n", fname1);
    return EXIT_FAILURE;
  }

  if (!f2) {
    fprintf(stderr, "Failed to open file for reading: %s\n", fname2);
    return EXIT_FAILURE;
  }

  Real err = 0;

  for (ptrdiff_t i = 0; i < N; ++i) {
    Comp c1, c2;
    if (!f1.read((char*) &c1, sizeof(Comp))) {
      fprintf(stderr, "Failed to read %zu bytes from %s\n",
        sizeof(Comp), fname1);
      return EXIT_FAILURE;
    }
    if (!f2.read((char*) &c2, sizeof(Comp))) {
      fprintf(stderr, "Failed to read %zu bytes from %s\n",
        sizeof(Comp), fname2);
      return EXIT_FAILURE;
    }
    err = max(err, abs(c1 - c2));
  }

  printf("Error: %g\n", err);

  return EXIT_SUCCESS;
}
