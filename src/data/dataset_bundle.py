import os
from dataclasses import dataclass

import pandas as pd


@dataclass
class DatasetSplit:
    x: pd.DataFrame
    y: pd.Series

@dataclass
class DatasetBundle:
    train: DatasetSplit
    val: DatasetSplit | None
    test: DatasetSplit

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        splits = [('train', self.train), ('test', self.test)]
        if self.val is not None:
            splits.append(('val', self.val))
        for split_name, split in splits:
            split.x.to_parquet(f"{path}/{split_name}_x.parquet", index=False)
            split.y.to_frame(name='target').to_parquet(f"{path}/{split_name}_y.parquet", index=False)

    @staticmethod
    def load(path: str, use_validation: bool) -> 'DatasetBundle':
        def load_split(name):
            x = pd.read_parquet(f"{path}/{name}_x.parquet")
            y_df = pd.read_parquet(f"{path}/{name}_y.parquet")
            y = y_df.iloc[:, 0]
            y.name = y_df.columns[0]
            return DatasetSplit(x, y)

        val=None
        if use_validation:
            val = load_split('val')

        return DatasetBundle(
            train=load_split('train'),
            val=val,
            test=load_split('test')
        )

