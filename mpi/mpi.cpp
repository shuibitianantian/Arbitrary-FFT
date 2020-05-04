#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

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
  fprintf(stderr, "MPI Runtime Error: %d\n", res);
  fprintf(stderr, "  At line %ld: %s\n", line, expr);
  exit(EXIT_FAILURE);
}
#endif

#include "common.h"

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
  const ptrdiff_t lOff;
  const ptrdiff_t lN;
  const ptrdiff_t lM;
  const int L;
  const int lId;
  Comp* const W;
  Comp* const B;
  Comp* const C;
  Comp* const Z;

  static Bluestein plan(ptrdiff_t N, bool sign, int lId, int np) {
    auto M = ceil2(N + N - 1);
    auto L = __builtin_ctzll(M);
    auto lM = M / np;
    auto lN = lM;
    auto lOff = lM * lId;
    if (lOff >= N)
      lN = 0;
    else if (lOff + lN > N)
      lN = N - lOff;
    return {sign, N, M, lOff, lN, lM, L, lId};
  }

  Bluestein(bool sign, ptrdiff_t N, ptrdiff_t M,
    ptrdiff_t lOff, ptrdiff_t lN, ptrdiff_t lM, int L, int lId) :
    N(N), M(M), lOff(lOff), lN(lN), lM(lM), L(L), lId(lId),
    W(alloc<Comp>(L)), B(alloc<Comp>(lM)), C(alloc<Comp>(lN)),
    Z(alloc<Comp>(lM))
  {
    for (int i = 0; i < L; ++i)
      W[i] = cis(ldexp(_2Pi, -i - 1));

    auto w = Pi / (Real) N;
    if (!sign)
      w = -w;

    auto beg1 = lOff - lOff;
    auto end1 = min(N, lOff + lN) - lOff;
    auto beg2 = max(N, lOff) - lOff;
    auto end2 = min(M - N + 1, lOff + lM) - lOff;
    auto beg3 = max(M - N + 1, lOff) - lOff;
    auto end3 = min(M, lOff + lM) - lOff;

    for (auto i = beg1; i < end1; ++i) {
      auto j = i + lOff;
      C[i] = cis(w * (Real) (j * j));
    }

    for (auto i = beg1; i < end1; ++i) {
      auto j = i + lOff;
      B[i] = cis(-w * (Real) (j * j));
    }

    if (beg2 < end2)
      fill(B + beg2, B + end2, Comp{});

    for (auto i = beg3; i < end3; ++i) {
      auto j = M - i - lOff;
      B[i] = cis(-w * (Real) (j * j));
    }

    dft(B);
  }

  ~Bluestein() {
    free(W);
    free(B);
    free(C);
    free(Z);
  }

  ptrdiff_t getSize(ptrdiff_t& off, ptrdiff_t& n) {
    off = lOff;
    n = lN;
    if (off >= N)
      off = 0;
    return lM;
  }

  void transform(Comp* Y) {
    auto beg1 = lOff - lOff;
    auto end1 = min(N, lOff + lN) - lOff;
    auto beg2 = max(N, lOff) - lOff;
    auto end2 = min(M, lOff + lM) - lOff;

    for (ptrdiff_t i = beg1; i < end1; ++i)
      Y[i] *= C[i];

    if (beg2 < end2)
      fill(Y + beg2, Y + end2, Comp{});

    dft(Y);

    for (ptrdiff_t i = 0; i < lM; ++i)
      Y[i] *= B[i];

    idft(Y);

    for (ptrdiff_t i = beg1; i < end1; ++i)
      Y[i] *= C[i];
  }

  void dft(Comp* Y) const {
    auto logM = __builtin_ctzll(lM);
    for (int i = L - 1; i >= logM; --i) {
      if (!(lOff >> i & 1)) {
        // lower
        checkMpi(MPI_Sendrecv(Y, lM, MpiComp, lId + (1 << (i - logM)), 0,
          Z, lM, MpiComp, lId + (1 << (i - logM)), 0,
          MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        for (ptrdiff_t k = 0; k < lM; ++k)
          Y[k] += Z[k];
      }
      else {
        // upper
        checkMpi(MPI_Sendrecv(Y, lM, MpiComp, lId - (1 << (i - logM)), 0,
          Z, lM, MpiComp, lId - (1 << (i - logM)), 0,
          MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        auto h = (ptrdiff_t) 1 << i;
        auto wn = W[i];
        auto w = cis(ldexp(_2Pi, -i - 1) * (lOff & (h - 1)));
        for (ptrdiff_t k = 0; k < lM; ++k) {
          Y[k] = w * (Z[k] - Y[k]);
          w *= wn;
        }
      }
    }
    for (int i = logM - 1; i >= 0; --i) {
      auto wn = W[i];
      auto h = (ptrdiff_t) 1 << i;
      for (ptrdiff_t j = 0; j < lM; j += h << 1) {
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
    auto logM = __builtin_ctzll(lM);
    for (int i = 0; i < logM; ++i) {
      auto wn = conj(W[i]);
      auto h = (ptrdiff_t) 1 << i;
      for (ptrdiff_t j = 0; j < lM; j += h << 1) {
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
    for (int i = logM; i < L; ++i) {
      if (!(lOff >> i & 1)) {
        // lower
        checkMpi(MPI_Sendrecv(Y, lM, MpiComp, lId + (1 << (i - logM)), 0,
          Z, lM, MpiComp, lId + (1 << (i - logM)), 0,
          MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        auto h = (ptrdiff_t) 1 << i;
        auto wn = conj(W[i]);
        auto w = cis(ldexp(-_2Pi, -i - 1) * (lOff & (h - 1)));
        for (ptrdiff_t k = 0; k < lM; ++k) {
          Y[k] += w * Z[k];
          w *= wn;
        }
      }
      else {
        // upper
        checkMpi(MPI_Sendrecv(Y, lM, MpiComp, lId - (1 << (i - logM)), 0,
          Z, lM, MpiComp, lId - (1 << (i - logM)), 0,
          MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        auto h = (ptrdiff_t) 1 << i;
        auto wn = conj(W[i]);
        auto w = cis(ldexp(-_2Pi, -i - 1) * (lOff & (h - 1)));
        for (ptrdiff_t k = 0; k < lM; ++k) {
          Y[k] = Z[k] - w * Y[k];
          w *= wn;
        }
      }
    }
    for (ptrdiff_t i = 0; i < lM; ++i)
      //Y[i] = Y[i].ldexp(-L);
      Y[i] /= (Real) M;
  }
};

int main(int nArg, char* args[]) {
  checkMpi(MPI_Init(&nArg, &args));
  int lId, np;
  checkMpi(MPI_Comm_rank(MPI_COMM_WORLD, &lId));
  checkMpi(MPI_Comm_size(MPI_COMM_WORLD, &np));

  bool sign;
  ptrdiff_t N;
  char *fnamei, *fnameo;
  if (!checkArgs(sign, N, fnamei, fnameo, nArg, args, lId) && !lId) {
    fprintf(stderr, "Usage: %s <sign> <N> <Input> <Output>\n", args[0]);
    checkMpi(MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE));
    return EXIT_FAILURE;
  }

  MPI_File fi, fo;
  MPI_Status iostat;
  int iocount;

  checkMpi(MPI_File_open(MPI_COMM_WORLD, fnamei,
    MPI_MODE_RDONLY, MPI_INFO_NULL, &fi));

  checkMpi(MPI_File_open(MPI_COMM_WORLD, fnameo,
    MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fo));

  if (!lId)
    printf("[Root] Start planning...\n");
  auto fft = Bluestein::plan(N, sign, lId, np);
  ptrdiff_t lOff, lN;
  auto lM = fft.getSize(lOff, lN);

  auto lMpiOff = MPI_Offset(sizeof(Comp) * lOff);

#if 0
  // START DEBUG OUT: B, C
  printf("%d: good\n", lId);

  auto fMpiOff = MPI_Offset(sizeof(Comp) * fft.lOff);

  MPI_File fb, fc;
  checkMpi(MPI_File_open(MPI_COMM_WORLD, "mb.dat",
    MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fb));
  checkMpi(MPI_File_open(MPI_COMM_WORLD, "mc.dat",
    MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fc));

  checkMpi(MPI_File_write_at_all(fb, fMpiOff, fft.B, fft.lM, MpiComp, &iostat));
  checkMpi(MPI_Get_count(&iostat, MpiComp, &iocount));
  if (iocount != fft.lM) {
    fprintf(stderr, "[Rank %d] Failed to store %zd complex numbers at "
      "offset %zd (only wrote %d)\n", lId, fft.lM, fft.lOff, iocount);
    checkMpi(MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE));
    return EXIT_FAILURE;
  }
  checkMpi(MPI_File_close(&fb));

  printf("%d: good bo\n", lId);

  checkMpi(MPI_File_write_at_all(fc, lMpiOff, fft.C, lN, MpiComp, &iostat));
  checkMpi(MPI_Get_count(&iostat, MpiComp, &iocount));
  if (iocount != lN) {
    fprintf(stderr, "[Rank %d] Failed to store %zd complex numbers at "
      "offset %zd (only wrote %d)\n", lId, lN, lOff, iocount);
    checkMpi(MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE));
    return EXIT_FAILURE;
  }
  checkMpi(MPI_File_close(&fc));

  printf("%d: good co\n", lId);

  checkMpi(MPI_Finalize());
  return EXIT_SUCCESS;
  // END DEBUG OUT: B, C
#endif

  auto Y = alloc<Comp>(lM);

  if (!lId)
    printf("[Root] Start loading data...\n");

  checkMpi(MPI_File_read_at_all(fi, lMpiOff, Y, lN, MpiComp, &iostat));
  checkMpi(MPI_Get_count(&iostat, MpiComp, &iocount));
  if (iocount != fft.lN) {
    fprintf(stderr, "[Rank %d] Failed to load %zd complex numbers at "
      "offset %zd (only got %d)\n", lId, lN, lOff, iocount);
    checkMpi(MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE));
    return EXIT_FAILURE;
  }
  checkMpi(MPI_File_close(&fi));

  if (!lId)
    printf("[Root] FFT starts\n");

  checkMpi(MPI_Barrier(MPI_COMM_WORLD));
  auto tStart = MPI_Wtime();

  fft.transform(Y);
  if (sign) {
    // Does not divide the result by N
    auto inv = 1 / (Real) N;
    for (ptrdiff_t i = 0; i < lN; ++i)
      Y[i] *= inv;
  }

  checkMpi(MPI_Barrier(MPI_COMM_WORLD));
  auto tEnd = MPI_Wtime();
  auto time = tEnd - tStart;

  if (!lId) {
    printf("[Root] FFT Ends\n");
    printf("[Root] Time: %f s\n", time);
    printf("[Root] Start saving data...\n");
  }

  checkMpi(MPI_File_write_at_all(fo, lMpiOff, Y, lN, MpiComp, &iostat));
  checkMpi(MPI_Get_count(&iostat, MpiComp, &iocount));
  if (iocount != lN) {
    fprintf(stderr, "[Rank %d] Failed to store %zd complex numbers at "
      "offset %zd (only wrote %d)\n", lId, lN, lOff, iocount);
    checkMpi(MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE));
    return EXIT_FAILURE;
  }

  checkMpi(MPI_File_close(&fo));

  if (!lId)
    printf("[Root] All done\n");

  checkMpi(MPI_Finalize());
  return EXIT_SUCCESS;
}
