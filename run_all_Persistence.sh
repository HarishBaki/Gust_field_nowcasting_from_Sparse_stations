#!/bin/bash

# Define arrays
data_types=("NYSM" "RTMA")
zarr_stores=("data/NYSM.zarr" "data/RTMA.zarr")
freqs=("5min" "60min")

# Loop over combinations
for i in "${!data_types[@]}"; do
    data_type="${data_types[$i]}"
    zarr_store="${zarr_stores[$i]}"
    freq="${freqs[$i]}"

    if [[ "$data_type" == "NYSM" ]]; then
        horizon_max=72
    else
        horizon_max=6
    fi
    for output_sequence_length in $(seq 1 $horizon_max); do
        sbatch --job-name="persist_${data_type}_${output_sequence_length}" \
            --export=data_type="${data_type}",zarr_store="${zarr_store}",freq="${freq}",output_sequence_length="${output_sequence_length}" \
            jobsub_Persistence.slurm
    done
done