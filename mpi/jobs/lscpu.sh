#!/bin/bash
#
#SBATCH --job-name=lscpu
#SBATCH --nodes=2
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=8GB
#SBATCH --output=lscpu-%j.out

module load openmpi/gnu/4.0.2
mpiexec lscpu
