from dataclasses import dataclass
from typing import List

@dataclass
class MLPConfig:
    hidden_dims: List[int]

@dataclass
class LSTMConfig:
    hidden_dim: int
    ff_hidden_dim: int

@dataclass
class ModelConfig:
    input_dim: int
    mlp: MLPConfig
    lstm: LSTMConfig
    output_dim: int

@dataclass
class TrainingConfig:
    dataset: str
    test_size: float
    val_size: float
    random_state: int
    epochs: int
    batch_size: int
    learning_rate: float
    eval_frequency: float
    route_feature_names: List[str]
    time_feature_names: List[str]


@dataclass
class Config:
    project_name: str
    model: ModelConfig
    training: TrainingConfig
    train_mlp: bool
    train_lstm: bool