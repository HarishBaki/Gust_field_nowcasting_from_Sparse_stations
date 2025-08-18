#!/bin/bash

#SBATCH --job-name=jupyter
#SBATCH --output=slurmout/jupyter-%j.out
#SBATCH --error=slurmout/jupyter-%j.err
#SBATCH --time=01-00:00:00
# #SBATCH --mem=80gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --container-name=physicsnemo
#SBATCH --container-mounts=/network/rit/dgx/dgx_basulab/Harish:/mnt/dgx_basulab/Harish,/network/rit/lab/basulab:/mnt/basulab,/network/rit/home/hb533188:/mnt/home/hb533188,/network/rit/dgx/dgx_basulab/Harish/Gust_field_nowcasting_from_Sparse_stations:/mnt/current_project,/network/rit/lab/basulab/Harish/Gust_field_nowcasting_from_Sparse_stations/data/MRMS_grib_data:/mnt/current_project/data/MRMS_grib_data
#SBATCH --container-workdir=/mnt/current_project

# (Optional) faster pip installs by using your home cache:
export PIP_CACHE_DIR=/mnt/home/hb533188/.cache/pip

# --- CDO via micromamba (user install) ---
export MAMBA_ROOT_PREFIX="/mnt/home/hb533188/.micromamba"
if [ ! -x "$MAMBA_ROOT_PREFIX/bin/micromamba" ]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /tmp bin/micromamba
  mkdir -p "$MAMBA_ROOT_PREFIX/bin"
  mv /tmp/bin/micromamba "$MAMBA_ROOT_PREFIX/bin/micromamba"
fi
eval "$("$MAMBA_ROOT_PREFIX/bin/micromamba" shell hook -s bash)"
if ! micromamba env list | grep -q '^cdo\s'; then
  micromamba create -y -n cdo -c conda-forge cdo eccodes libnetcdf netcdf4 hdf5 udunits gdal proj
fi
micromamba activate cdo
cdo -V

# Install required Python packages inside the container
pip install --quiet timm torchmetrics seaborn metpy herbie-data basemap basemap-data-hires lxml

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
