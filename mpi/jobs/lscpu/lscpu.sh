#!/bin/bash
#
#SBATCH --job-name=lscpu
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:01:00
#SBATCH --mem=16MB
#SBATCH --output=lscpu-%j.out

module load openmpi/gnu/3.1.4
mpiexec hostname
mpiexec lscpu
