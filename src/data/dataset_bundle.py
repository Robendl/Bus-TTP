from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class DatasetSplit:
    x: pd.DataFrame
    y: pd.DataFrame

@dataclass
class DatasetBundle:
    train: DatasetSplit
    val: DatasetSplit
    test: DatasetSplit
    # route_lookup: Dict[str, np.array]
    # aggr_route_lookup: Dict[str, np.array]
