#!/bin/bash

#SBATCH --time=0-04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120GB
#SBATCH --job-name=xgb_ttp
#SBATCH --output=xgb.out

module purge
module load Python/3.13

source ~/venvs/bus-ttp/bin/activate

python3 src/main.py --config-name config_habrok train_mlp=False train_lstm=False fit_xgboost=True compute_baseline=True use_validation=True

deactivate
