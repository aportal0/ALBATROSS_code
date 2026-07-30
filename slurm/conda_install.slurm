#!/bin/bash
#SBATCH --job-name=install_conda
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --output=install_conda_%j.out
#SBATCH --error=install_conda_%j.err

source ~/anaconda3/etc/profile.d/conda.sh

conda create -n climate-indices python=3.11 -y
conda activate climate-indices
pip install --upgrade pip setuptools wheel
pip install climate-indices

