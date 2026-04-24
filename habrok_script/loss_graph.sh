#!/bin/bash

#SBATCH --time=1-00:00:00
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

python3 src/main.py --config-name config_habrok build_dataset=True
python3 src/main.py --config-name config_habrok training.random_state=12
python3 src/main.py --config-name config_habrok training.random_state=13
python3 src/main.py --config-name config_habrok training.random_state=14
python3 src/main.py --config-name config_habrok training.random_state=15
python3 src/main.py --config-name config_habrok training.random_state=16

deactivate
