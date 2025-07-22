#!/bin/bash

#SBATCH --time=0-00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=38GB
#SBATCH --job-name=bus-ttp
#SBATCH --output=bus-ttp.out

module purge
module load Python/3.13

source ~/venvs/bus-ttp/bin/activate

python3 src/heatmap.py --config-name config_habrok

deactivate
