#!/bin/bash

#SBATCH --time=0-00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --job-name=data-conv
#SBATCH --output=data-conv.out

module purge
module load Python/3.13

source ~/venvs/bus-ttp/bin/activate

python3 src/analysis.py --config-name config_habrok

deactivate
