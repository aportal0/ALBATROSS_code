#!/bin/bash
#SBATCH --job-name=install_xesmf
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --output=install_xesmf_%j.out
#SBATCH --error=install_xesmf_%j.err

source ~/anaconda3/etc/profile.d/conda.sh

# conda create -n xesmf_env
conda activate xesmf_env

# conda install -c conda-forge xesmf
# conda install -c conda-forge dask netCDF4
conda install -c conda-forge matplotlib cartopy
