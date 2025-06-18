from dataclasses import dataclass
from typing import List

@dataclass
class MLPConfig:
    hidden_dims: List[int]

@dataclass
class ModelConfig:
    input_dim: int
    mlp: MLPConfig
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


@dataclass
class Config:
    project_name: str
    model: ModelConfig
    training: TrainingConfig
    train: bool