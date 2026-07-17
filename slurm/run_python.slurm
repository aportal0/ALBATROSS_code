#!/bin/bash
#SBATCH --job-name=python
#SBATCH --output=python_%j.out
#SBATCH --error=python_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Load modules
module purge
module load atmos/cdo/2.3.0

# Activate your Python environment
conda activate myenv

# Run job
python 24h_mean.py
