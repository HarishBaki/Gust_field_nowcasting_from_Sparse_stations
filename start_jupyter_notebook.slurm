#!/bin/bash

#SBATCH --job-name=jupyter
#SBATCH --output=slurmout/jupyter-%j.out
#SBATCH --error=slurmout/jupyter-%j.err
#SBATCH --time=01-00:00:00
# #SBATCH --mem=80gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=256
#SBATCH --container-image=docker://nvcr.io/nvidia/physicsnemo/physicsnemo:25.03
#SBATCH --container-name=physicsnemo
#SBATCH --container-mounts=/network/rit/dgx/dgx_basulab/Harish:/mnt/dgx_basulab/Harish,/network/rit/lab/basulab:/mnt/basulab,/network/rit/home/hb533188:/mnt/home/hb533188,/network/rit/dgx/dgx_basulab/Harish/Gust_field_nowcasting_from_Sparse_stations:/mnt/current_project,/network/rit/home/hb533188:/mnt/home/hb533188,/network/rit/lab/basulab/Harish/Gust_field_nowcasting_from_Sparse_stations/Predictions:/mnt/current_project/Predictions,/network/rit/lab/basulab/Harish/Gust_field_nowcasting_from_Sparse_stations/data/MRMS_grib_data:/mnt/current_project/data/MRMS_grib_data,/network/rit/lab/basulab/Harish/Downloads/s5cmd_2.3.0_Linux-64bit:/mnt/s5cmd
#SBATCH --container-workdir=/mnt/current_project

export PYTHONNOUSERSITE=1   # make Python use the system site-packages, not from local user

# Install required Python packages inside the container
echo "Installing packages..."
pip install --quiet --no-deps timm torchmetrics seaborn herbie-data lxml metpy pyproj lpips scikit-image --prefix=/tmp/python
echo "Done installing packages"

export PYTHONPATH=/tmp/python/local/lib/python3.12/site-packages:/tmp/python/local/lib/python3.12/dist-packages:$PYTHONPATH  # I already know that the container python is 3.12


export PATH=$PATH:/mnt/s5cmd

# Get the DGX node name
node_name="$SLURMD_NODENAME"

# Extract node number from hostname, assuming it's like 'dgx01', 'dgx02', etc.
node_number=$(echo "$node_name" | grep -oP '\d+$' | awk '{printf "%02d", $1}')

echo -e "Jupyter Lab is being loaded..."

# Generate a random port number between 8000 and 8999
port=$((RANDOM % 1000 + 8000))

password='Harish'

# Build the Jupyter URL
#jupyter_url="http://${node_name}.its.albany.edu:${port}"
jupyter_url="http://169.226.129.1${node_number}:${port}"
jupyter_url="${jupyter_url}/lab?token=${password}"

# Print session details
echo -e "Your Jupyter Lab session will be available at: ${jupyter_url}\n"
echo -e "Your password is: ${password}\n"
echo -e "Please copy and paste the link into your browser and use the password to log in.\n"
echo -e "================================================================================\n"

# Start Jupyter Lab session
jupyter lab --allow-root --no-browser --NotebookApp.token="${password}" --NotebookApp.allow_origin='*' --NotebookApp.log_level='CRITICAL' --notebook-dir=/mnt/current_project --port=$port
