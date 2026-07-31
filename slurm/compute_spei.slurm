#!/bin/bash
#SBATCH --job-name=spei
#SBATCH --output=logs/spei_%j.out
#SBATCH --error=logs/spei_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

# Activate python3 environment
source /home/PERSONALE/alice.portal2/anaconda3/etc/profile.d/conda.sh
conda activate climate-indices
python -c "import distutils; print(distutils.__file__)"

mkdir -p "/home/PERSONALE/alice.portal2/scratch/ERA5-Land/SPEI/"

process_climate_indices \
  --index spei \
  --periodicity monthly \
  --netcdf_precip "/home/PERSONALE/alice.portal2/scratch/ERA5-Land/ET0/monthly/precip_monthly_1993-2024_Madagascar_new1.nc" \
  --var_name_precip "precipitation" \
  --netcdf_pet "/home/PERSONALE/alice.portal2/scratch/ERA5-Land/ET0/monthly/ET0_Mod-Hargreaves_monthly_1993-2024_Madagascar_new1.nc" \
  --var_name_pet "ET0" \
  --scales 1 3 6 12 \
  --calibration_start_year 1993 \
  --calibration_end_year 2024 \
  --output_file_base "/home/PERSONALE/alice.portal2/scratch/ERA5-Land/SPEI/SPEI_Mod-Hargreaves_1-3-6-12m_1993-2024_Madagascar.nc" 

