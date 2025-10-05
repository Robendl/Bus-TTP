#!/bin/bash

#SBATCH --time=0-04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=54GB
#SBATCH --job-name=pfi-ttp
#SBATCH --output=pfi-ttp.out

module purge
module load Python/3.13

source ~/venvs/bus-ttp/bin/activate

python3 src/main.py --config-name config_habrok pre_data_conversions=True
python3 src/feature_importance_lstm.py --config-name config_habrok

deactivate
