#!/bin/bash

#SBATCH --time=0-04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128GB
#SBATCH --job-name=data-conv
#SBATCH --output=data-conv.out

module purge
module load Python/3.13

source ~/venvs/bus-ttp/bin/activate

python3 src/main_feature_selection.py --config-name config_habrok

deactivate
