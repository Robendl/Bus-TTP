"""Lightweight (de)serialization wrapper around the train/val/test splits."""
import os
from dataclasses import dataclass

import pandas as pd

from config.config import Config


@dataclass
class DatasetSplit:
    x: pd.DataFrame
    y: pd.Series


@dataclass
class DatasetBundle:
    train: DatasetSplit
    val: DatasetSplit | None
    test: DatasetSplit | None

    @staticmethod
    def _path_for(base_path: str, cfg: Config) -> str:
        return (
            base_path
            + ("_val" if cfg.dataset.use_validation else "")
            + ("_pca" if cfg.dataset.pca else "")
            + ("_fulltrain" if not cfg.dataset.use_test else "")
            + ("_multi" if cfg.dataset.multi_run else "")
        )

    def save(self, path: str, cfg: Config) -> None:
        path = self._path_for(path, cfg)
        os.makedirs(path, exist_ok=True)

        splits = [("train", self.train)]
        if self.val is not None:
            splits.append(("val", self.val))
        if self.test is not None:
            splits.append(("test", self.test))

        for name, split in splits:
            split.x.to_parquet(f"{path}/{name}_x.parquet", index=False)
            split.y.to_frame(name="target").to_parquet(f"{path}/{name}_y.parquet", index=False)

    @staticmethod
    def load(path: str, cfg: Config) -> "DatasetBundle":
        path = DatasetBundle._path_for(path, cfg)

        def load_split(name: str) -> DatasetSplit:
            x = pd.read_parquet(f"{path}/{name}_x.parquet")
            y_df = pd.read_parquet(f"{path}/{name}_y.parquet")
            y = y_df.iloc[:, 0]
            y.name = y_df.columns[0]
            return DatasetSplit(x, y)

        return DatasetBundle(
            train=load_split("train"),
            val=load_split("val") if cfg.dataset.use_validation else None,
            test=load_split("test") if cfg.dataset.use_test else None,
        )
