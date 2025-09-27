#!/bin/bash
#SBATCH --job-name=MRMS
#SBATCH --output=slurmout/MRMS-%A_%a.out
#SBATCH --error=slurmout/MRMS-%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6

# --- Activate conda ---
source /network/rit/lab/basulab/Anaconda3/etc/profile.d/conda.sh
conda activate conus404

# ============================================================
# Arguments:
#   $1 = START_DATE or missing_dates file
#   $2 = MODE (optional: "single" | "array")
#
# Usage examples:
#   sbatch jobsub_conus_download.slurm 1980-07-15 single
#   sbatch --array=0-364 jobsub_conus_download.slurm 1980-01-01 array
# ============================================================

INPUT=$1
MODE=${2:-array}   # default = array mode

if [ -z "$INPUT" ]; then
    echo "Usage: $0 START_DATE|missing_dates.txt [single|array|file]"
    exit 1
fi

# --- Determine DATE based on mode ---
if [ "$MODE" == "single" ]; then
    # Run for exactly one day
    DATE=$INPUT

elif [ "$MODE" == "array" ]; then
    # Serial incrementing with START_DATE + offset, formatted as YYYYMMDD
    DATE=$(date -d "${INPUT} +${SLURM_ARRAY_TASK_ID} days" +%Y%m%d)

else
    echo "Invalid mode: $MODE"
    echo "Valid modes: single | array "
    exit 1
fi

echo "=========================================================="
echo "Running MRMS daily processing"
echo "Mode: $MODE"
echo "Date: $DATE"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "=========================================================="

python Process_and_write_to_zarr_MRMS_daily.py --data_type MergedReflectivityQCComposite_00.50 --day "$DATE"