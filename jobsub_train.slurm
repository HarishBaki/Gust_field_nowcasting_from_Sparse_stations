#!/bin/bash

#SBATCH --job-name=ddp_train
#SBATCH --output=slurmout/ddp_train-%j.out
#SBATCH --error=slurmout/ddp_train-%j.err
#SBATCH --time=07-00:00:00
#SBATCH --mem=320gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --exclude=dgx02
#SBATCH --container-image=docker://nvcr.io/nvidia/physicsnemo/physicsnemo:25.03
#SBATCH --container-name=physicsnemo
#SBATCH --container-mounts=/network/rit/dgx/dgx_basulab/Harish:/mnt/dgx_basulab/Harish,/network/rit/lab/basulab:/mnt/basulab,/network/rit/home/hb533188:/mnt/home/hb533188,/network/rit/dgx/dgx_basulab/Harish/Gust_field_nowcasting_from_Sparse_stations:/mnt/current_project,/network/rit/lab/basulab/Harish/Gust_field_nowcasting_from_Sparse_stations/data/MRMS_grib_data:/mnt/current_project/data/MRMS_grib_data,/network/rit/lab/basulab/Harish/Downloads/s5cmd_2.3.0_Linux-64bit:/mnt/s5cmd
#SBATCH --container-workdir=/mnt/current_project
# #SBATCH --container-image=/home/harish/softwares/container_images/physicsnemo:25.03.sqsh
# #SBATCH --container-mounts=/home/harish:/home/harish,/home/harish/Ongoing_Research/Gust_field_nowcasting_from_Sparse_stations:/workspace,/data/harish/Gust_field_nowcasting_from_Sparse_stations:/workspace/data
# #SBATCH --container-workdir=/workspace

# Optionally pass SLURM values into the shell script as environment variables
export nproc_per_node=${SLURM_NTASKS_PER_NODE}
export num_workers=${SLURM_CPUS_PER_TASK}
export MASTER_PORT=$((20000 + RANDOM % 20000))

# ===============================
# Set defaults if not provided
# ===============================
variable=${variable:-'i10fg'}
checkpoint_dir=${checkpoint_dir:-'checkpoints'}
model_name=${model_name:-'predrnn_v2'}
activation_layer=${activation_layer:-'gelu'}
transform=${transform:-'none'}
batch_size=${batch_size:-12}
num_workers=${num_workers:-16}
weights_seed=${weights_seed:-42}
num_epochs=${num_epochs:-200}
loss_name=${loss_name:-'MaskedCharbonnierLoss'}
resume=${resume:-'false'}
input_window_size=${input_window_size:-36}
output_window_size=${output_window_size:-36}
step_size=${step_size:-1}
forecast_offset=${forecast_offset:-0}

# ===============================
# Echo all config values
# ===============================
echo "======================================"
echo "Launching training with configuration:"
echo "--------------------------------------"
echo "Variable:                 $variable"
echo "Checkpoint Dir:           $checkpoint_dir"
echo "Model:                    $model_name"
echo "Activation Layer:         $activation_layer"
echo "Transform:                $transform"
echo "Batch Size:               $batch_size"
echo "Num Workers:              $num_workers"
echo "Weights Seed:             $weights_seed"
echo "Num Epochs:               $num_epochs"
echo "Loss Function:            $loss_name"
echo "Resume from Checkpoint:   $resume"
echo "Input Window Size:        $input_window_size"
echo "Output Window Size:       $output_window_size"
echo "Step Size:                $step_size"
echo "Forecast Offset:          $forecast_offset"
echo "======================================"

export PYTHONNOUSERSITE=1   # make Python use the system site-packages, not from local user
# Install required Python packages inside the container
echo "Installing packages..."
pip install --quiet --no-deps timm torchmetrics seaborn herbie-data lxml metpy pyproj lpips scikit-image --prefix=/tmp/python
echo "Done installing packages"
export PYTHONPATH=/tmp/python/local/lib/python3.12/site-packages:/tmp/python/local/lib/python3.12/dist-packages:$PYTHONPATH  # I already know that the container python is 3.12

# ===============================
# Run the training script
# ===============================
torchrun --nproc_per_node=$nproc_per_node --master_port=$MASTER_PORT train_PredRNN-V2.py \
  --variable "$variable" \
  --checkpoint_dir "$checkpoint_dir" \
  --model_name "$model_name" \
  --activation_layer "$activation_layer" \
  --transform "$transform" \
  --batch_size "$batch_size" \
  --num_workers "$num_workers" \
  --weights_seed "$weights_seed" \
  --num_epochs "$num_epochs" \
  --loss_name "$loss_name" \
  --input_window_size "$input_window_size" \
  --output_window_size "$output_window_size" \
  --step_size "$step_size" \
  --forecast_offset "$forecast_offset" \
  $( [ "$resume" = "true" ] && echo "--resume" )