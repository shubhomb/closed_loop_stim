#!/bin/bash
#SBATCH --job-name=optimal-learning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --threads-per-core=1
#SBATCH --mem-per-cpu=1024M
#SBATCH --time=04:00:00
#SBATCH --export=ALL
#SBATCH --output=tmp/optimal_learning_%j.out

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

ml purge
ml Mamba/23.11.0-0
source activate /scratch/qj3/qj3-pytorch-2.4.1-cuda-12.4-July9-2025

srun bash -lc "$1"