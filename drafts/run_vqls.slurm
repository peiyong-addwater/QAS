#!/bin/bash --login
#SBATCH --job-name=vqls-test
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --account=pawsey0419
#SBATCH --export=None

Export OMP_NUM_THREADS=24

module load singularity

