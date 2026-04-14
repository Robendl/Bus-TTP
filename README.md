# Bus Travel Time Prediction for Unseen Routes

Master's thesis project for Artificial Intelligence at the University of Groningen, conducted with Irias Information Management.

Full thesis report: [thesis-report.pdf](thesis-report.pdf)

## Why this project matters

Most bus travel-time models rely on dense GPS trajectory data and known routes. This project tackles a harder, practical setting: predicting travel times for unseen origin-destination (OD) pairs using only static trip attributes and road-network characteristics.

The objective is decision support for planners and schedulers who want reliable time estimates before new or modified routes are deployed.

## What I built

- End-to-end machine learning pipeline in Python for data conversion, cleaning, feature engineering, training, evaluation, and analysis.
- Leakage-aware OD-pair disjoint train/test strategy to evaluate generalization to unseen OD pairs.
- Comparative modeling framework with four approaches:
  - Linear Regression baseline
  - XGBoost
  - Multilayer Perceptron (MLP)
  - LSTM + feedforward hybrid model
- Experiment tooling for:
  - Bootstrap confidence intervals and significance tests
  - Ablation studies
  - PCA efficiency/performance trade-off analysis
  - Feature importance with both PFI and SHAP
  - Error diagnostics (REC curves, residual and distribution analysis, geospatial heatmaps)
- Export path for deployment artifacts (ONNX conversion for preprocessing and neural models).

## Data and methodology

The dataset combines:

- Trip-level operational data (from project partner context)
- Dutch National Road Files (NWB) attributes
- OpenStreetMap-derived road characteristics
- Temporal/context features (cyclical encodings, holidays, school vacations)

The codebase includes robust preprocessing steps such as filtering invalid/implausible entries, route-level outlier handling, scaling, and optional PCA. Mapping quality is explicitly evaluated before model training.

## Results (from thesis evaluation)

Main test-set performance (bootstrapped by OD-pair, 95% CI):

- **Linear Regression:** MAE 24.56, MAPE 32.25, RMSE 27.64
- **XGBoost:** MAE 18.25, MAPE 18.44, RMSE 21.44
- **MLP:** MAE 16.49, MAPE 16.73, RMSE 19.98
- **LSTM:** MAE 16.39, MAPE 16.50, RMSE 19.81

Key findings:

- Neural models clearly outperformed non-neural baselines.
- LSTM performed slightly better than MLP, but the difference was not statistically significant (paired test p = 0.240).
- At a tolerance of +/-20 seconds, accuracy ranged from 61.75% (Linear Regression) to 79.19% (LSTM).
- Feature importance consistently highlighted **distance**, **segment length**, and **max speed** as dominant predictors.
- Pipeline quality mattered: skipping critical cleaning steps (especially implausible speed filtering) caused major performance degradation.

## Tech stack

- Python, PyTorch, XGBoost, scikit-learn
- pandas, NumPy, pyarrow/fastparquet
- SHAP, matplotlib, seaborn
- Hydra/OmegaConf configuration system

## Repository structure

- `src/main.py`: main experiment entry point
- `src/data/`: preprocessing, dataset construction, dataloaders
- `src/model/`: MLP and LSTM architectures
- `src/train/`: training and evaluation logic
- `src/plot/`: plotting and analysis utilities
- `config/`: experiment configuration files

## Reproducibility notes

- Configuration is centralized through Hydra (`config/config.yaml` and variants).
- Outputs (metrics, artifacts, plots, model files) are organized by run directory.
- The repository expects local datasets under the configured `datasets` paths.
