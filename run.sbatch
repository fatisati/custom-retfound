#!/bin/bash
#SBATCH --job-name=run                 # Job name
#SBATCH --output=./out_files/run_out.txt            # Standard output log
#SBATCH --error=./out_files/run_err.txt            # Standard error log
#SBATCH --partition=cpu_p                   # Specify the partition
#SBATCH --qos=cpu_long
#SBATCH --nodes=1                              # Number of nodes
#SBATCH --ntasks=1                             # Number of tasks (processes)
#SBATCH --cpus-per-task=1                      # Number of CPU cores per task
#SBATCH --mem=1G                             # Request 150 GB of memory

# Load required modules (if any, e.g., Python, Anaconda)
# module load anaconda3

# Activate the virtual environment
# source /home/e-helmholtz/lodechan/bin/activate  # Adjust the path to your virtual environment
source $HOME/.bashrc
export PYTHONUNBUFFERED=1

python experiment_runner.py
