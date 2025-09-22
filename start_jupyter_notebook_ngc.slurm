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
#SBATCH --container-image=docker://nvcr.io/nvidia/physicsnemo/physicsnemo:25.03
#SBATCH --container-name=physicsnemo
#SBATCH --container-mounts=/home/hb533188/dgx_basulab/Harish/Gust_field_nowcasting_from_Sparse_stations:/mnt/current_project
#SBATCH --container-workdir=/mnt/current_project

export PYTHONNOUSERSITE=1   # make Python use the system site-packages, not from local user

# Install required Python packages inside the container
echo "Installing packages..."
pip install --quiet --no-deps timm torchmetrics seaborn herbie-data lxml metpy pyproj lpips scikit-image --prefix=/tmp/python
echo "Done installing packages"

export PYTHONPATH=/tmp/python/local/lib/python3.12/site-packages:/tmp/python/local/lib/python3.12/dist-packages:$PYTHONPATH  # I already know that the container python is 3.12

# Get the DGX node name (e.g., gpu001, gpu002)
node_name="$SLURMD_NODENAME"

# Choose a random port between 8000–8999 for Jupyter
port=$((RANDOM % 1000 + 8000))

password='Harish'

echo "======================================================================"
echo " Jupyter Lab is starting on node: $node_name"
echo " Remote port inside job: $port"
echo ""

echo ""
echo " To connect from your local machine or from vscode, first you need to do port forwarding."
echo " Open a terminal and run:"
echo ""
echo "   ssh -L 8890:${node_name}:${port} dgx-cloud"
echo ""
echo " This forwards remote port ${port} on $node_name to local port 8890."
echo ""
echo " Then, in your local browser, open:"
echo ""
echo "   http://127.0.0.1:8890/?token=${password}"
echo ""
echo " (You can also open http://127.0.0.1:8890/lab?token=${password} directly in a browser.)"
echo ""
echo "To connect in vscode, you need to use the original dgx-cloud node link:"
echo ""
echo " http://${node_name}:${port}/?token=${password}"
echo ""
echo "======================================================================"

# Start Jupyter Lab session
jupyter lab --allow-root --no-browser --NotebookApp.token="${password}" --NotebookApp.allow_origin='*' --NotebookApp.log_level='CRITICAL' --notebook-dir=/mnt/current_project --port=$port
