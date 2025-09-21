#!/bin/bash

#SBATCH --job-name=rclone
#SBATCH --output=slurmout/rclone-%j.out
#SBATCH --error=slurmout/rclone-%j.err
#SBATCH --time=01-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

# Add your rclone binary location to PATH
export PATH=$PATH:/network/rit/lab/basulab/Harish/Downloads/rclone-v1.67.0-linux-amd64

rclone copy --progress --transfers 12 data/NYSM.zarr ngc:/home/hb533188/dgx_basulab/Harish/Gust_field_nowcasting_from_Sparse_stations/data/NYSM.zarr