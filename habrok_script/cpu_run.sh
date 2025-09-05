#!/bin/bash

#SBATCH --time=0-04:00:00
#SBATCH --ntasks=1
#SBATCH --mem=70GB
#SBATCH --job-name=data-conv
#SBATCH --output=data-conv.out

module purge
module load Python/3.13

source ~/venvs/bus-ttp/bin/activate

python3 src/main.py --config-name config_habrok pre_data_conversions=True

deactivate
