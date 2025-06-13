#!/bin/bash

#SBATCH --time=0-00:10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --job-name=bus-ttp
#SBATCH --output=bus-ttp.out

module purge
module load Python/3.13

source ~/venvs/bus-ttp/bin/activate

python3 src/main.py --config-name config_habrok

deactivate
