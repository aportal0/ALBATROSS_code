#!/bin/bash
#SBATCH --job-name=python
#SBATCH --output=logs/python_%j.out
#SBATCH --error=logs/python_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

mkdir -p logs
HOST=$(hostname -s)

# Directory
if [[ "${HOST:0:3}" == "ac6" ]]; then
    # Directory
    dir_code="/home/ecme4047/ALBATROSS_code/python/"
    # Activate your Python environment
    module load conda
    conda activate xesmf_env
elif [[ "$HOST" == "ophfe1" ]]; then 
    # Directory
    dir_code="/home/PERSONALE/alice.portal2/ALBATROSS_code/python/"
    # Load modules
    module purge
    module load atmos/cdo/2.3.0
    # Activate your Python environment
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate xesmf_env
fi

# Run job
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
python ${dir_code}compute_spei.py
