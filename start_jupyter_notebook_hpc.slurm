#!/bin/bash

#SBATCH --job-name=jupyter
#SBATCH --output=slurmout/jupyter-%j.out
#SBATCH --error=slurmout/jupyter-%j.err
#SBATCH --time=01-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256

# Load conda init script
source /network/rit/lab/basulab/Anaconda3/etc/profile.d/conda.sh
conda activate conus404

export PATH=$PATH:/mnt/s5cmd

# Random port
port=$((RANDOM % 1000 + 8000))
token="Harish"

# Detect node hostname/IP
node_host=$(hostname -f)

# Print URL to .out
echo "=========================================================="
echo "Your Jupyter Lab session is available at:"
echo "   http://${node_host}:${port}/lab?token=${token}"
echo "=========================================================="

# Start Jupyter Lab
jupyter lab --no-browser --ip=0.0.0.0 --port=$port \
  --NotebookApp.token="${token}"