#include <complex>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <random>

using namespace std;

#include <mpi.h>

#define Strify_(s_) # s_
#define Strify(s_) Strify_(s_)

#ifdef NDEBUG
#define checkMpi(e_) (e_)
#else
#define checkMpi(e_) (implCheckMpi((e_), __LINE__, Strify(e_)))
void implCheckMpi(int res, long line, const char* expr) {
  if (res == MPI_SUCCESS)
    return;
  std::fprintf(stderr, "MPI Runtime Error: %d\n", res);
  std::fprintf(stderr, "  At line %ld: %s\n", line, expr);
  std::exit(EXIT_FAILURE);
}
#endif

using Real = double;
using Comp = complex<Real>;
#define MpiComp MPI_C_DOUBLE_COMPLEX

constexpr int N = 100;
constexpr char fname[] = "n100.dat";

Comp A[N], B[N];

int main(int nArg, char* args[]) {
  checkMpi(MPI_Init(&nArg, &args));

  auto fp = fopen(fname, "rb");
  for (auto i = 0; i < N; ++i)
    if (sizeof(Comp) != fread(A + i, 1, sizeof(Comp), fp)) {
      fprintf(stderr, "stdio failed\n");
      return EXIT_FAILURE;
    }
  fclose(fp);

  MPI_File mf;
  MPI_Status st;
  checkMpi(MPI_File_open(MPI_COMM_WORLD, fname, MPI_MODE_RDONLY, MPI_INFO_NULL,
    &mf));
  checkMpi(MPI_File_read_at(mf, 0, B, N, MpiComp, &st));
  int count;
  checkMpi(MPI_Get_count(&st, MpiComp, &count));
  if (count != N) {
    fprintf(stderr, "mpi failed\n");
  }

  if (memcmp(A, B, sizeof(A)))
    fprintf(stderr, "discrepancy\n");
  else
    printf("ok\n");

  checkMpi(MPI_Finalize());
  return 0;
}
