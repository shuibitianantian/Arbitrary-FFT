#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

struct Version {
  string prefix;
  string exec;
  vector<string> module;
};

Version vers[]{
  {"fftw", "../fftw-mpi", {"fftw/openmpi/intel/3.3.5"}},
  {"mpih", "../mpi-h", {"openmpi/gnu/3.1.4"}},
};

constexpr ptrdiff_t Ns[]{
  //100000, 1000003,
  //1000000, 1000003,
  10000000, 10000019,
  //100000000, 100000007,
};

constexpr int nTaskPerNode = 16;
constexpr int nNode = 16;

int main() {
  auto user = getenv("USER");
  auto nMaxTask = nTaskPerNode * nNode;
  ofstream gen("gen.sh");
  ofstream all("all.sh");
  gen << "rm -fr /scratch/" << user << "/*\n";
  for (auto N : Ns) {
    ostringstream oss;
    oss << "/scratch/" << user << "/n" << N << "-in.dat";
    auto finame = oss.str();
    gen << "../gen " << N << " " << finame << "\n";
    for (auto np = 1; np <= nMaxTask; np <<= 1) {
      auto nodes = max(1, np / nTaskPerNode);
      auto tasksPerNode = min(np, nTaskPerNode);
      ostringstream oss;
      oss << "n" << N << "-np" << np;
      auto name = oss.str();
      all << "sbatch " << name << ".sh\n";
      ofstream job(name + ".sh");
      job << "#!/bin/bash\n";
      job << "#\n";
      job << "#SBATCH --job-name=" << name << "\n";
      job << "#SBATCH --nodes=" << nodes << "\n";
      job << "#SBATCH --tasks-per-node=" << tasksPerNode << "\n";
      job << "#SBATCH --cpus-per-task=1\n";
      job << "#SBATCH --time=00:20:00\n";
      job << "#SBATCH --mem=8GB\n";
      job << "#SBATCH --partition=c01_17\n";
      //if (np > 1)
      //  job << "#SBATCH --partition=c26\n";
      //else
      //  job << "#SBATCH --partition=c18_25\n";
      job << "#SBATCH --output=" << name << ".out\n";
      //job << "#SBATCH --error=" << name << ".err\n";
      for (auto sign = 0; sign < 2; ++sign) {
        vector<string> ss;
        for (auto& ver : vers) {
          job << "\n";
          job << "module purge\n";
          for (auto& mod : ver.module)
            job << "module load " << mod << "\n";
          ostringstream oss;
          oss << "/scratch/" << user << "/" << name;
          oss << "-" << ver.prefix << "-s" << sign << ".dat";
          auto foname = oss.str();
          job << "echo vvvvvvvvvvvvvvvv " << foname << " vvvvvvvvvvvvvvvv\n";
          job << "mpiexec " << ver.exec << " " << sign << " " << N << " ";
          job << finame << " " << foname << "\n";
          job << "echo ^^^^^^^^^^^^^^^^ " << foname << " ^^^^^^^^^^^^^^^^\n";
          job << "echo\n";
          ss.emplace_back(move(foname));
        }
        for (auto i = 0; i < ss.size(); ++i)
          for (auto j = i + 1; j < ss.size(); ++j) {
            job << "\n";
            job << "echo ======== CMP " << ss[i] << " ";
            job << ss[j] << " ========\n";
            job << "../cmp " << N << " " << ss[i] << " " << ss[j] << "\n";
            job << "echo\n";
          }
        job << "\n";
        for (auto& s : ss)
          job << "rm -f " << s << "\n";
      }
    }
  }
  return 0;
}
