#!/bin/bash
#
#SBATCH --job-name=n10000019-np2
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=8GB
#SBATCH --output=n10000019-np2.out

module purge
module load openmpi/gnu/4.0.2
echo vvvvvvvvvvvvvvvv n10000019-np2-mpih-s0 vvvvvvvvvvvvvvvv
mpiexec ../mpi-h 0 10000019 n10000019-in.dat n10000019-np2-mpih-s0.dat
echo ^^^^^^^^^^^^^^^^ n10000019-np2-mpih-s0 ^^^^^^^^^^^^^^^^

module purge
module load fftw/openmpi/intel/3.3.5
echo vvvvvvvvvvvvvvvv n10000019-np2-fftw-s0 vvvvvvvvvvvvvvvv
mpiexec ../fftw-mpi 0 10000019 n10000019-in.dat n10000019-np2-fftw-s0.dat
echo ^^^^^^^^^^^^^^^^ n10000019-np2-fftw-s0 ^^^^^^^^^^^^^^^^

echo ======== CMP n10000019-np2-mpih-s0.dat n10000019-np2-fftw-s0.dat ========
../cmp 10000019 n10000019-np2-mpih-s0.dat n10000019-np2-fftw-s0.dat
echo

rm -f n10000019-np2-mpih-s0.dat
rm -f n10000019-np2-fftw-s0.dat

module purge
module load openmpi/gnu/4.0.2
echo vvvvvvvvvvvvvvvv n10000019-np2-mpih-s1 vvvvvvvvvvvvvvvv
mpiexec ../mpi-h 1 10000019 n10000019-in.dat n10000019-np2-mpih-s1.dat
echo ^^^^^^^^^^^^^^^^ n10000019-np2-mpih-s1 ^^^^^^^^^^^^^^^^

module purge
module load fftw/openmpi/intel/3.3.5
echo vvvvvvvvvvvvvvvv n10000019-np2-fftw-s1 vvvvvvvvvvvvvvvv
mpiexec ../fftw-mpi 1 10000019 n10000019-in.dat n10000019-np2-fftw-s1.dat
echo ^^^^^^^^^^^^^^^^ n10000019-np2-fftw-s1 ^^^^^^^^^^^^^^^^

echo ======== CMP n10000019-np2-mpih-s1.dat n10000019-np2-fftw-s1.dat ========
../cmp 10000019 n10000019-np2-mpih-s1.dat n10000019-np2-fftw-s1.dat
echo

rm -f n10000019-np2-mpih-s1.dat
rm -f n10000019-np2-fftw-s1.dat
