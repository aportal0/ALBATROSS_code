#!/bin/bash

module load atmos/cdo/2.3.0

indir="/ec/res4/scratch/ecme4047/ERA5-Land/t2m/daily"
outdir="/ec/res4/scratch/ecme4047/ERA5-Land/t2m/monthly"

mkdir -p "$outdir"

stats="minimum"

for year in $(seq 2021 2021); do

    echo "Processing $year"

    files=$(find "${indir}/${year}" -type f -name "*${stats}*.nc" | sort)
    
    cdo -O monmean \
        -mergetime $files \
        "$outdir/t2m_${stats}_monthly_${year}.nc"

done
