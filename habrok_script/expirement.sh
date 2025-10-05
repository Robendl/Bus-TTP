#!/bin/bash

#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=56GB
#SBATCH --job-name=pre
#SBATCH --output=pre-bus.out

module purge
module load Python/3.13

source ~/venvs/bus-ttp/bin/activate

python3 src/main.py --config-name config_habrok pre_data_conversions=True dataset.scale_features=False
python3 src/main.py --config-name config_habrok dataset.scale_features=False

python3 src/main.py --config-name config_habrok pre_data_conversions=True dataset.filter_outliers=False
python3 src/main.py --config-name config_habrok dataset.filter_outliers=False

python3 src/main.py --config-name config_habrok pre_data_conversions=True dataset.include_mapping_errors=True
python3 src/main.py --config-name config_habrok dataset.include_mapping_errors=True

python3 src/main.py --config-name config_habrok pre_data_conversions=True dataset.include_measurement_errors=True
python3 src/main.py --config-name config_habrok dataset.include_measurement_errors=True

python3 src/main.py --config-name config_habrok pre_data_conversions=True dataset.include_invalid=True
python3 src/main.py --config-name config_habrok dataset.include_invalid=True

python3 src/main.py --config-name config_habrok pre_data_conversions=True dataset.pca=True dataset.n_components=0.4
python3 src/main.py --config-name config_habrok dataset.pca=True dataset.n_components=0.4

python3 src/main.py --config-name config_habrok pre_data_conversions=True dataset.pca=True dataset.n_components=0.6
python3 src/main.py --config-name config_habrok dataset.pca=True dataset.n_components=0.6

python3 src/main.py --config-name config_habrok pre_data_conversions=True dataset.pca=True dataset.n_components=0.8
python3 src/main.py --config-name config_habrok dataset.pca=True dataset.n_components=0.8

python3 src/main.py --config-name config_habrok pre_data_conversions=True dataset.pca=True dataset.n_components=0.95
python3 src/main.py --config-name config_habrok dataset.pca=True dataset.n_components=0.95

deactivate
