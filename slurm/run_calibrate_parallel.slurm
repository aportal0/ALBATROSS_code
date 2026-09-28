#!/bin/bash
#SBATCH --job-name=drydays
#SBATCH --account=spitport
#SBATCH --qos=nf
#SBATCH --partition=gpil
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/calibrate_%j.out
#SBATCH --error=logs/calibrate_%j.err

set -euo pipefail

# ---- environment -----------------------------------------------------
module purge
module load prgenv/intel          # or: prgenv/gnu, cdo, etc.
module load python3               # ECMWF's python3 module
# module load netcdf4 hdf5 ...    # whatever your site requires

# dask reads these from the environment at worker start
export DASK_DISTRIBUTED__WORKER__MEMORY__TARGET=0.80
export DASK_DISTRIBUTED__WORKER__MEMORY__SPILL=0.85
export DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE=0.90

# keep the thread pools from oversubscribing the node
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ---- run -------------------------------------------------------------
PYTHON_DIR="/home/ecme4047/ALBATROSS_code/python"
srun --cpu-bind=cores python3 ${PYTHON_DIR}/calibrate_seasfc_to_obs_parallel.py

