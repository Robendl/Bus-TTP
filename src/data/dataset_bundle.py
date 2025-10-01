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

    def save(self, path: str, cfg: Config):
        path = (path
                + ("_val" if cfg.dataset.use_validation else "")
                + ("_pca" if cfg.dataset.pca else "")
                + ("_fulltrain" if cfg.dataset.use_test else "")
                + ("_multi" if cfg.dataset.multi_run else ""))
        os.makedirs(path, exist_ok=True)
        splits = [('train', self.train)]
        if self.val is not None:
            splits.append(('val', self.val))
        if self.test is not None:
            splits.append(('test', self.test))
        for split_name, split in splits:
            split.x.to_parquet(f"{path}/{split_name}_x.parquet", index=False)
            split.y.to_frame(name='target').to_parquet(f"{path}/{split_name}_y.parquet", index=False)

    @staticmethod
    def load(path: str, cfg: Config) -> 'DatasetBundle':
        path = (path
                + ("_val" if cfg.dataset.use_validation else "")
                + ("_pca" if cfg.dataset.pca else "")
                + ("_fulltrain" if not cfg.dataset.use_test else "")
                + ("_multi" if cfg.dataset.multi_run else ""))
        def load_split(name):
            x = pd.read_parquet(f"{path}/{name}_x.parquet")
            y_df = pd.read_parquet(f"{path}/{name}_y.parquet")
            y = y_df.iloc[:, 0]
            y.name = y_df.columns[0]
            return DatasetSplit(x, y)

        train = load_split('train')
        val=None
        test=None
        if cfg.dataset.use_validation:
            val = load_split('val')

        if cfg.dataset.use_test:
            test = load_split('test')

        return DatasetBundle(
            train=train,
            val=val,
            test=test
        )

