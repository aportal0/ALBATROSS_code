#!/bin/bash

module load atmos/cdo/2.3.0

indir="/home/PERSONALE/alice.portal2/scratch/MSWEP/MSWEP_V316_test/Past/Daily"
outdir="/home/PERSONALE/alice.portal2/scratch/MSWEP/MSWEP_V316_test/Past/Monthly"

mkdir -p "$outdir"

for year in $(seq 2000 2000); do

    echo "Processing $year"

    cdo -O monsum \
        -mergetime $(ls "$indir"/${year}[0-9][0-9][0-9].nc | sort) \
        "$outdir/${year}.nc"

done
