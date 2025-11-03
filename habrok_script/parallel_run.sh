#!/bin/bash

#SBATCH --time=0-04:00:00
#SBATCH --partition=parallel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --job-name=fs
#SBATCH --output=fs.out

module purge
module load Python/3.13

source ~/venvs/bus-ttp/bin/activate

python3 src/feature_selection.py --config-name config_habrok

deactivate
