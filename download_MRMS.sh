#!/bin/bash

# Parameters
BUCKET="s3://noaa-mrms-pds"
OUTDIR="data/MRMS_grib_data"

START=20201014
END=20201014

# Loop by day (uses GNU date; on mac: gdate)
d="$START"
while [ "$d" -le "$END" ]; do
    echo ">> $d"
    # --- For reflectivity at the lowest altitude ---
    # PRODUCT="CONUS/MergedReflectivityAtLowestAltitude_00.50"
    # mkdir -p "$OUTDIR/$PRODUCT/$d"
    # s5cmd --no-sign-request --numworkers 32 cp "$BUCKET/$PRODUCT/$d/*.grib2.gz" "$OUTDIR/$PRODUCT/$d/"

    # --- For Merged composite reflectivity quality controlled ---
    PRODUCT="CONUS/MergedReflectivityQCComposite_00.50"
    
    mkdir -p "$OUTDIR/$PRODUCT/$d"
    s5cmd --no-sign-request --numworkers 32 sync "$BUCKET/$PRODUCT/$d/*.grib2.gz" "$OUTDIR/$PRODUCT/$d/"

    d=$(date -d "$d + 1 day" +%Y%m%d)
done