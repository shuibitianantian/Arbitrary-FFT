#!/bin/bash
#
#SBATCH --job-name=lscpu
#SBATCH --nodes=8
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:01:00
#SBATCH --mem=256MB
#SBATCH --output=lscpu-%j.out

module load openmpi/gnu/4.0.2
mpiexec hostname
mpiexec lscpu
