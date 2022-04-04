#!/bin/bash --login
#SBATCH --job-name=vqls-test
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --account=pawsey0419
#SBATCH --export=None

Export OMP_NUM_THREADS=24

module load singularity
srun --export=all -N 1 -n 1 -c 1 singularity exec --home CURRENT_DIRECTORY_FULL quantum-research-220404.sif python drafts/vqls_test.py