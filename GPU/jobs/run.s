#!/bin/bash

##SBATCH --nodes=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=hpc
#SBATCH --mail-type=END
##SBATCH --mail-user=yx1919@nyu.edu
#SBATCH --output=output.out


module purge
module load cuda/10.1.105

nvcc -std=c++11 -o main ../run_benchmark.cu -O3 -lcufft  -Xcompiler -fopenmp

./main 10
./main 100
./main 1000
./main 10000
./main 100000
./main 1000000
./main 10000000

echo "Testing prime numbers"
echo

./main 11
./main 101
./main 1009
./main 10007
./main 100003
./main 1000003
./main 10000019
