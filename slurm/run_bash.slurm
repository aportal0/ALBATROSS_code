#!/bin/bash
#SBATCH --job-name=python
#SBATCH --output=cdo_%j.out
#SBATCH --error=cdo_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Directory
dir_code="/home/PERSONALE/alice.portal2/ALBATROSS_code/bash/"

# Load modules
module purge
module load atmos/cdo/2.3.0

# Run job
bash ${dir_code}monsum_MSWEP.sh
