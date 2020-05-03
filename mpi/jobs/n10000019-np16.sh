#!/bin/bash
#
#SBATCH --job-name=n10000019-np16
#SBATCH --nodes=2
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=8GB
#SBATCH --partition=c26
#SBATCH --output=n10000019-np16.out

module purge
module load fftw/openmpi/intel/3.3.5
echo vvvvvvvvvvvvvvvv /scratch/zl2972/n10000019-np16-fftw-s0.dat vvvvvvvvvvvvvvvv
mpiexec ../fftw-mpi 0 10000019 /scratch/zl2972/n10000019-in.dat /scratch/zl2972/n10000019-np16-fftw-s0.dat
echo ^^^^^^^^^^^^^^^^ /scratch/zl2972/n10000019-np16-fftw-s0.dat ^^^^^^^^^^^^^^^^
echo

module purge
module load openmpi/gnu/3.1.4
echo vvvvvvvvvvvvvvvv /scratch/zl2972/n10000019-np16-mpih-s0.dat vvvvvvvvvvvvvvvv
mpiexec ../mpi-h 0 10000019 /scratch/zl2972/n10000019-in.dat /scratch/zl2972/n10000019-np16-mpih-s0.dat
echo ^^^^^^^^^^^^^^^^ /scratch/zl2972/n10000019-np16-mpih-s0.dat ^^^^^^^^^^^^^^^^
echo

echo ======== CMP /scratch/zl2972/n10000019-np16-fftw-s0.dat /scratch/zl2972/n10000019-np16-mpih-s0.dat ========
../cmp 10000019 /scratch/zl2972/n10000019-np16-fftw-s0.dat /scratch/zl2972/n10000019-np16-mpih-s0.dat
echo

rm -f /scratch/zl2972/n10000019-np16-fftw-s0.dat
rm -f /scratch/zl2972/n10000019-np16-mpih-s0.dat

module purge
module load fftw/openmpi/intel/3.3.5
echo vvvvvvvvvvvvvvvv /scratch/zl2972/n10000019-np16-fftw-s1.dat vvvvvvvvvvvvvvvv
mpiexec ../fftw-mpi 1 10000019 /scratch/zl2972/n10000019-in.dat /scratch/zl2972/n10000019-np16-fftw-s1.dat
echo ^^^^^^^^^^^^^^^^ /scratch/zl2972/n10000019-np16-fftw-s1.dat ^^^^^^^^^^^^^^^^
echo

module purge
module load openmpi/gnu/3.1.4
echo vvvvvvvvvvvvvvvv /scratch/zl2972/n10000019-np16-mpih-s1.dat vvvvvvvvvvvvvvvv
mpiexec ../mpi-h 1 10000019 /scratch/zl2972/n10000019-in.dat /scratch/zl2972/n10000019-np16-mpih-s1.dat
echo ^^^^^^^^^^^^^^^^ /scratch/zl2972/n10000019-np16-mpih-s1.dat ^^^^^^^^^^^^^^^^
echo

echo ======== CMP /scratch/zl2972/n10000019-np16-fftw-s1.dat /scratch/zl2972/n10000019-np16-mpih-s1.dat ========
../cmp 10000019 /scratch/zl2972/n10000019-np16-fftw-s1.dat /scratch/zl2972/n10000019-np16-mpih-s1.dat
echo

rm -f /scratch/zl2972/n10000019-np16-fftw-s1.dat
rm -f /scratch/zl2972/n10000019-np16-mpih-s1.dat
