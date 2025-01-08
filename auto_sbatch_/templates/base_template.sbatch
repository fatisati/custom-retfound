#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_file}
#SBATCH --error={error_file}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=150G
#SBATCH --account=core-kind
#SBATCH --gres=gpu:1

source ~/.bashrc
export HOME=/data/core-kind/fatemeh

conda activate retfound

