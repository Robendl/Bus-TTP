from dataclasses import dataclass
from typing import List

@dataclass
class MLPConfig:
    dropout: float
    hidden_dim: int
    hidden_dims: List[int]

@dataclass
class LSTMConfig:
    bidirectional: bool
    dropout: float
    lstm_hidden_dim: int
    ff_hidden_dims: List[int]
    num_lstm_layers: int

@dataclass
class XgboostConfig:
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    reg_lambda: float
    reg_alpha: float
    num_boost_round: int

@dataclass
class ModelConfig:
    input_dim: int
    mlp: MLPConfig
    lstm: LSTMConfig
    xgboost: XgboostConfig
    output_dim: int

@dataclass
class OptimizerConfig:
    type: str
    learning_rate: float
    weight_decay: float
    scheduler: str

@dataclass
class TrainingConfig:
    test_size: float
    val_size: float
    random_state: int
    epochs: int
    batch_size: int
    eval_frequency: float
    patience: int
    min_delta: float
    scheduler: str
    early_stopping_enabled: bool
    route_based_training: bool
    optimizer_mlp: OptimizerConfig
    optimizer_lstm: OptimizerConfig

@dataclass
class DatasetConfig:
    iqr_factor: float
    time: str
    route_seq: str
    route_aggr: str
    metadata: str
    multi_run: bool
    geoms: str
    use_subset: bool
    scale_features: bool
    use_test: bool
    use_validation: bool
    scaling_route_features: List[str]
    scaling_time_features: List[str]
    route_feature_names: List[str]
    time_feature_names: List[str]
    route_feature_names_full: List[str]
    time_feature_names_full: List[str]
    pca: bool
    n_components: float
    filter_outliers: bool
    include_mapping_errors: bool
    include_measurement_errors: bool
    include_invalid: bool
    process_metadata: bool
    residual_plot_features: List[str]

@dataclass
class PlotConfig:
    margins_max: int
    percentages_max: int
    step_size: int

@dataclass
class Config:
    project_name: str
    model: ModelConfig
    training: TrainingConfig
    compute_baseline: bool
    train_mlp: bool
    train_lstm: bool
    fit_xgboost: bool
    pre_data_conversions: bool
    dataset: DatasetConfig
    plot: PlotConfig
    save_results: bool