#!/bin/sh
module purge
module load fftw/openmpi/intel/3.3.5
export CXXFLAGS=-I/share/apps/fftw/3.3.5/openmpi/intel/include/
export LDFLAGS=-L/share/apps/fftw/3.3.5/openmpi/intel/lib/
make fftw-mpi

module purge
module load gcc/9.1.0
module load openmpi/gnu/4.0.2
make
