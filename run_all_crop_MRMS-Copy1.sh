#!/bin/bash
# conda activate cdo
# Parameters
OUTDIR="data/MRMS_grib_data"
TARGET_GRID="data/orography_grid_cf.nc"

START=20240715
END=20240716
MAXJOBS=64

d="$START"
while [ "$d" -le "$END" ]; do
    echo ">> Processing $d"

    for PRODUCT in \
        "CONUS/MergedReflectivityAtLowestAltitude_00.50" \
        "CONUS/MergedReflectivityQCComposite_00.50"
    do
        echo "   -> $PRODUCT"

        SRCDIR="$OUTDIR/$PRODUCT/$d/source"
        UNZIPDIR="$OUTDIR/$PRODUCT/$d/unzipped"
        CROPDIR="$OUTDIR/$PRODUCT/$d/cropped_NYS"

        mkdir -p "$UNZIPDIR" "$CROPDIR"

        njobs=0

        # --- Step 1: unzip ---
        for f in "$SRCDIR"/*.grib2.gz; do
            [ -f "$f" ] || continue
            fname=$(basename "$f" .gz)
            gunzip -c "$f" > "$UNZIPDIR/$fname" &

            ((njobs++))
            if (( njobs % MAXJOBS == 0 )); then
                wait
            fi
        done
        wait

        njobs=0

        # --- Step 2: interpolate ---
        for f in "$UNZIPDIR"/*.grib2; do
            [ -f "$f" ] || continue
            fname=$(basename "$f" .grib2)
            cdo -f nc4c remapbil,"$TARGET_GRID" "$f" "$CROPDIR/${fname}_on_orog.nc" &

            ((njobs++))
            if (( njobs % MAXJOBS == 0 )); then
                wait
            fi
        done
        wait

        # --- Step 3: cleanup ---
        echo "   -> Cleaning up $SRCDIR"
        #rm -rf "$SRCDIR"
        
    done
    
    d=$(date -d "$d + 1 day" +%Y%m%d)
done