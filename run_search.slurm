#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Sat Jul 17 2021 08:50:42 GMT+1000 (Australian Eastern Standard Time)

# Partition for the job:
#SBATCH --partition=gpgpu

# Multithreaded (SMP) job: must run on one node
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="CQAS"

# The project ID which this job should run under:
#SBATCH --account="punim0619"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

# Number of GPUs requested per node:
#SBATCH --gres=gpu:1
# The amount of memory in megabytes per process in the job:
#SBATCH --mem=4000

# Use this email address:
#SBATCH --mail-user=addwater0315@gmail.com

# Send yourself an email when the job:
# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# begins
#SBATCH --mail-type=BEGIN
# ends successfully
#SBATCH --mail-type=END

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=5-0:0:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)

# The modules to load:
module load anaconda3/2020.11
module load cuda/11.1.1
# module load web_proxy/1.0
# module load git/2.28.0-nodocs
# module load cmake/3.20.1
# git clone https://github.com/qulacs/qulacs.git
# cd qulacs
# python setup_gpu.py install --user

# The job command(s):
python search_sth_tree_search.py