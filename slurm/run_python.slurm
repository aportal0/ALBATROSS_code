#!/bin/bash
#SBATCH --job-name=python
#SBATCH --output=logs/python_%j.out
#SBATCH --error=logs/python_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Directory
dir_code="/home/PERSONALE/alice.portal2/ALBATROSS_code/python/"

# Load modules
module purge
module load atmos/cdo/2.3.0

# Activate your Python environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate xesmf_env

# Run job
python ${dir_code}compute_et0.py
