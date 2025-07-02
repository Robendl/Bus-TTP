#!/bin/bash

#SBATCH --time=0-00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --job-name=bus-ttp
#SBATCH --output=bus-ttp.out

module purge
module load Python/3.13

source ~/venvs/bus-ttp/bin/activate

python3 src/main_lstm.py --config-name config_habrok

deactivate
