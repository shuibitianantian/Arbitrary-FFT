#include <algorithm>
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
  {"mpih", "../mpi-h", {"openmpi/gnu/4.0.2"}},
  {"fftw", "../fftw-mpi", {"fftw/openmpi/intel/3.3.5"}},
};

constexpr ptrdiff_t Ns[]{
  /*100000, 1000000,*/ 10000000, /*100000000,*/
  /*100003, 1000003,*/ 10000019, /*100000007,*/
};

constexpr int nTaskPerNode = 8;
constexpr int nNode = 8;

int main() {
  auto nMaxTask = nTaskPerNode * nNode;
  ofstream gen("gen.sh");
  ofstream all("all.sh");
  for (auto N : Ns) {
    ostringstream oss;
    oss << "n" << N << "-in.dat";
    auto iname = oss.str();
    gen << "../gen " << N << ' ' << iname << '\n';
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
      job << "#SBATCH --job-name=" << name << '\n';
      job << "#SBATCH --nodes=" << nodes << '\n';
      job << "#SBATCH --tasks-per-node=" << tasksPerNode << '\n';
      job << "#SBATCH --cpus-per-task=1\n";
      job << "#SBATCH --time=1:00:00\n";
      job << "#SBATCH --mem=8GB\n";
      job << "#SBATCH --partition=c18_25\n";
      job << "#SBATCH --output=" << name << ".out\n";
      job << "#SBATCH --error=" << name << ".err\n";
      for (auto sign = 0; sign < 2; ++sign) {
        vector<string> ss;
        for (auto& ver : vers) {
          job << "\n";
          job << "module purge\n";
          for (auto& mod : ver.module)
            job << "module load " << mod << '\n';
          ostringstream oss;
          oss << name << "-" << ver.prefix << "-s" << sign;
          auto oname = oss.str();
          job << "echo vvvvvvvvvvvvvvvv " << oname << " vvvvvvvvvvvvvvvv\n";
          job << "mpiexec " << ver.exec << ' ' << sign << ' ' << N << ' ';
          job << iname << ' ' << oname << ".dat" << '\n';
          job << "echo ^^^^^^^^^^^^^^^^ " << oname << " ^^^^^^^^^^^^^^^^\n";
          job << "echo\n";
          ss.emplace_back(move(oname));
        }
        for (auto i = 0; i < ss.size(); ++i)
          for (auto j = i + 1; j < ss.size(); ++j) {
            job << '\n';
            job << "echo ======== CMP " << ss[i] << ".dat ";
            job << ss[j] << ".dat ========\n";
            job << "../cmp " << N << ' ' << ss[i] << ".dat ";
            job << ss[j] << ".dat\n";
            job << "echo\n";
          }
        job << '\n';
        for (auto& s : ss)
          job << "rm -f " << s << ".dat\n";
      }
    }
  }
  return 0;
}
