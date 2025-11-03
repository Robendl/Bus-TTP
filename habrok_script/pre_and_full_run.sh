#!/bin/bash

#SBATCH --time=0-04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=70GB
#SBATCH --job-name=pre-bus
#SBATCH --output=pre-bus.out

module purge
module load Python/3.13

source ~/venvs/bus-ttp/bin/activate

python3 src/main.py --config-name config_habrok pre_data_conversions=True dataset.use_test=False
python3 src/full_train.py --config-name config_habrok dataset.use_test=False

deactivate
