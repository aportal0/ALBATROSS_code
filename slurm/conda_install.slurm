#!/bin/bash
#SBATCH --job-name=install_conda
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --output=install_conda_%j.out
#SBATCH --error=install_conda_%j.err

source ~/anaconda3/etc/profile.d/conda.sh

conda create -n xesmf_env --file ~/requirements_xesmf_env.txt

