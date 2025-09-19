#!/bin/bash

#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --job-name=grids
#SBATCH --output=grids.out

module purge
module load Python/3.13

source ~/venvs/bus-ttp/bin/activate

python3 src/gridsearch.py --config-name config_gs_old

deactivate
