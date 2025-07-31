from dataclasses import dataclass
from typing import List

@dataclass
class MLPConfig:
    hidden_dims: List[int]
    learning_rate: float

@dataclass
class LSTMConfig:
    hidden_dim: int
    ff_hidden_dim: int
    learning_rate: float

@dataclass
class ModelConfig:
    input_dim: int
    mlp: MLPConfig
    lstm: LSTMConfig
    output_dim: int

@dataclass
class TrainingConfig:
    test_size: float
    val_size: float
    random_state: int
    epochs: int
    batch_size: int
    eval_frequency: float

@dataclass
class DatasetConfig:
    time: str
    route_seq: str
    route_aggr: str
    metadata: str
    geoms: str
    route_feature_names: List[str]
    time_feature_names: List[str]


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
    pre_data_conversions: bool
    dataset: DatasetConfig
    plot: PlotConfig
    save_results: bool