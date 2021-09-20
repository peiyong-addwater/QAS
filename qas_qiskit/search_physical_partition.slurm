#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Fri Sep 17 2021 15:59:20 GMT+1000 (Australian Eastern Standard Time)

# Partition for the job:
#SBATCH --partition=physical

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="S_PHY_S"

# The project ID which this job should run under:
#SBATCH --account="punim0147"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1



# The amount of memory in megabytes per process in the job:
#SBATCH --mem=32768

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

# The job command(s):
python search_sth_tree_search.py