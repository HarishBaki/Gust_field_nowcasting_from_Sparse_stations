#!/bin/bash

# Loop through years 2018 to 2024
for YEAR in $(seq 2018 2024); do

    # figure out number of days in the year
    if date -d "$YEAR-02-29" >/dev/null 2>&1; then
        DAYS=366
    else
        DAYS=365
    fi

    START_DATE="${YEAR}0101"   # YYYYMMDD format

    echo "Submitting array job for YEAR=$YEAR with $DAYS days (start=$START_DATE)"

    # submit job array, passing YEAR and START_DATE to the job script
    sbatch --array=0-$((DAYS-1)) jobsub_process_MRMS.slurm $START_DATE 'array'

done