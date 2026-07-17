#!/bin/bash

module load atmos/cdo/2.3.0

indir="/home/PERSONALE/alice.portal2/scratch/ERA5-Land/t2m/daily"
outdir="/home/PERSONALE/alice.portal2/scratch/ERA5-Land/t2m/monthly"

mkdir -p "$outdir"

stats="maximum"

for year in $(seq 1993 2025); do

    echo "Processing $year"

    files=$(find "${indir}/${year}" -type f -name "*${stats}*.nc" | sort)
    
    cdo -O monmean \
        -mergetime $files \
        "$outdir/t2m_${stats}_monthly_${year}.nc"

done
