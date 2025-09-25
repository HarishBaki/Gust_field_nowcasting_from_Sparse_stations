#!/bin/bash

#SBATCH --job-name=persistence
#SBATCH --output=slurmout/persistence-%j.out
#SBATCH --error=slurmout/persistence-%j.err
#SBATCH --time=07-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
# #SBATCH --container-image=docker://nvcr.io/nvidia/physicsnemo/physicsnemo:25.03
# #SBATCH --container-name=physicsnemo
# #SBATCH --container-mounts=/network/rit/dgx/dgx_basulab/Harish:/mnt/dgx_basulab/Harish,/network/rit/lab/basulab:/mnt/basulab,/network/rit/home/hb533188:/mnt/home/hb533188,/network/rit/dgx/dgx_basulab/Harish/Gust_field_nowcasting_from_Sparse_stations:/mnt/current_project,/network/rit/lab/basulab/Harish/Gust_field_nowcasting_from_Sparse_stations/data/MRMS_grib_data:/mnt/current_project/data/MRMS_grib_data,/network/rit/lab/basulab/Harish/Downloads/s5cmd_2.3.0_Linux-64bit:/mnt/s5cmd
# #SBATCH --container-workdir=/mnt/current_project
# #SBATCH --container-image=/home/harish/softwares/container_images/physicsnemo:25.03.sqsh
# #SBATCH --container-name=physicsnemo
# #SBATCH --container-mounts=/home/harish:/home/harish,/home/harish/Ongoing_Research/Gust_field_nowcasting_from_Sparse_stations:/workspace,/data/harish/Gust_field_nowcasting_from_Sparse_stations:/workspace/data
# #SBATCH --container-workdir=/workspace

# === Activate Conda Environment ===
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gUstNET   # Replace with your actual env name

# Optionally pass SLURM values into the shell script as environment variables
export nproc_per_node=${SLURM_NTASKS_PER_NODE}
export num_workers=${SLURM_CPUS_PER_TASK}
export MASTER_PORT=$((20000 + RANDOM % 20000))

# ===============================
# Set defaults if not provided
# ===============================
prediction_dir=${prediction_dir:-'Predictions'}
data_type=${data_type:-'RTMA'}
zarr_store=${zarr_store:-'data/NYSM.zarr'}
start_date=${start_date:-'2024-01-01T00:00:00'}
end_date=${end_date:-'2024-12-31T23:59:59'}
freq=${freq:-'60min'}
output_sequence_length=${output_sequence_length:-72}
batch_size=${batch_size:-32}
num_workers=${num_workers:-16}

# ===============================
# Echo all config values
# ===============================
echo "======================================"
echo "Launching evaluation with configuration:"
echo "--------------------------------------"
echo "Prediction Dir:           $prediction_dir"
echo "Data Type:                $data_type"
echo "Zarr Store:               $zarr_store"
echo "Start Date:               $start_date"
echo "End Date:                 $end_date"
echo "Frequency:                $freq"
echo "Output Window Size:       $output_sequence_length"
echo "Batch Size:               $batch_size"
echo "Num Workers:              $num_workers"
echo "======================================"

# Install required Python packages inside the container
echo "Installing packages..."
pip install --quiet --no-deps timm torchmetrics seaborn herbie-data lxml metpy pyproj lpips scikit-image
echo "Done installing packages"

# ===============================
# Run the training script
# ===============================
torchrun --nproc_per_node=$nproc_per_node --master_port=$MASTER_PORT Persistence.py \
    --prediction_dir "$prediction_dir" \
    --data_type "$data_type" \
    --zarr_store "$zarr_store" \
    --start_date "$start_date" \
    --end_date "$end_date" \
    --freq "$freq" \
    --output_sequence_length $output_sequence_length \
    --batch_size $batch_size \
    --num_workers $num_workers
