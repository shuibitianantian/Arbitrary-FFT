#define NDEBUG

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <utility>

using namespace std;

#include <fftw3.h>
#include <fftw3-mpi.h>
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
  fprintf(stderr, "MPI Runtime Error: %d\n", res);
  fprintf(stderr, "  At line %ld: %s\n", line, expr);
  exit(EXIT_FAILURE);
}
#endif

using Real = double;
using Comp = fftw_complex;
#define MpiReal MPI_DOUBLE_PRECISION
#define MpiComp MPI_C_DOUBLE_COMPLEX

bool checkArgs(bool& sign, ptrdiff_t& N, char*& fnamei, char*& fnameo,
  int nArg, char* args[], int lId)
{
  if (nArg != 5) {
    if (!lId)
      fprintf(stderr, "Incorrect command line\n");
    return false;
  }
  int b;
  if (1 != sscanf(args[1], "%d", &b) || (b & ~1)) {
    if (!lId)
      fprintf(stderr, "Invalid sign: %s\n", args[1]);
    return false;
  }
  sign = b;
  if (1 != sscanf(args[2], "%zd", &N) || N < 1) {
    if (!lId)
      fprintf(stderr, "Invalid N: %s\n", args[1]);
    return false;
  }
  fnamei = args[3];
  fnameo = args[4];
  return true;
}


int main(int nArg, char* args[]) {
  checkMpi(MPI_Init(&nArg, &args));
  int lId, np;
  checkMpi(MPI_Comm_rank(MPI_COMM_WORLD, &lId));
  checkMpi(MPI_Comm_size(MPI_COMM_WORLD, &np));

  bool sign_;
  ptrdiff_t N;
  char *fnamei, *fnameo;
  if (!checkArgs(sign_, N, fnamei, fnameo, nArg, args, lId) && !lId) {
    fprintf(stderr, "Usage: %s <sign> <N> <Input> <Output>\n", args[0]);
    checkMpi(MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE));
    return EXIT_FAILURE;
  }
  auto sign = sign_ ? FFTW_BACKWARD : FFTW_FORWARD;

  MPI_File fi, fo;
  MPI_Status iostat;
  int iocount;

  checkMpi(MPI_File_open(MPI_COMM_WORLD, fnamei,
    MPI_MODE_RDONLY, MPI_INFO_NULL, &fi));

  checkMpi(MPI_File_open(MPI_COMM_WORLD, fnameo,
    MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fo));

  fftw_mpi_init();

  ptrdiff_t liN, liOff, loN, loOff;
  auto lN = fftw_mpi_local_size_1d(N, MPI_COMM_WORLD, sign, 0,
    &liN, &liOff, &loN, &loOff);

  auto liMpiOff = MPI_Offset(sizeof(Comp) * liOff);
  auto loMpiOff = MPI_Offset(sizeof(Comp) * loOff);

  auto X = fftw_alloc_complex(lN);
  auto Y = fftw_alloc_complex(lN);

  if (!lId)
    printf("[Root] Start planning...\n");
  auto plan = fftw_mpi_plan_dft_1d(N, X, Y, MPI_COMM_WORLD, sign,
    FFTW_ESTIMATE);

  if (!lId)
    printf("[Root] Start loading data...\n");

  //printf("[Rank %d] lN=%zd liOff=%zd liN=%zd\n", lId, lN, liOff, liN);

  checkMpi(MPI_File_read_at_all(fi, liMpiOff, X, liN, MpiComp, &iostat));
  checkMpi(MPI_Get_count(&iostat, MpiComp, &iocount));
  if (iocount != liN) {
    fprintf(stderr, "[Rank %d] Failed to load %zd complex numbers at "
      "offset %zd (only got %d)\n", lId, liN, liOff, iocount);
    checkMpi(MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE));
    return EXIT_FAILURE;
  }
  checkMpi(MPI_File_close(&fi));

  //for (ptrdiff_t i = 0; i < liN; ++i)
  //  printf("[Rank %d][%zd][%zd]: %g + i * %g\n", lId, i, liOff + i, X[i][0], X[i][1]);

  if (!lId)
    printf("[Root] FFT starts\n");

  checkMpi(MPI_Barrier(MPI_COMM_WORLD));
  auto tStart = MPI_Wtime();

  fftw_execute(plan);

  if (sign_) {
    // FFTW does not divide the result by N
    auto inv = 1 / (Real) N;
    for (ptrdiff_t i = 0; i < loN; ++i) {
      Y[i][0] *= inv;
      Y[i][1] *= inv;
    }
  }

  checkMpi(MPI_Barrier(MPI_COMM_WORLD));
  auto tEnd = MPI_Wtime();
  auto time = tEnd - tStart;

  if (!lId) {
    printf("[Root] FFT Ends\n");
    printf("[Root] Time: %f s\n", time);
    printf("[Root] Start saving data...\n");
  }

  checkMpi(MPI_File_write_at_all(fo, loMpiOff, Y, loN, MpiComp, &iostat));
  checkMpi(MPI_Get_count(&iostat, MpiComp, &iocount));
  if (iocount != loN) {
    fprintf(stderr, "[Rank %d] Failed to store %zd complex numbers at "
      "offset %zd (only wrote %d)\n", lId, loN, loOff, iocount);
    checkMpi(MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE));
    return EXIT_FAILURE;
  }
  checkMpi(MPI_File_close(&fo));

  fftw_free(X);
  fftw_free(Y);

  fftw_destroy_plan(plan);

  if (!lId)
    printf("[Root] All done\n");

  checkMpi(MPI_Finalize());
  return EXIT_SUCCESS;
}
