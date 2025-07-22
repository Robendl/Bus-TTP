import hydra
import torch

import config.paths as paths
import numpy as np
import pandas as pd

from config.config import Config
from data.data_conversions import load_route_lookup
from data.data_processing import create_dataloaders
from data.dataset_bundle import DatasetBundle
from data.mapping_dataset import aggr_collate_fn
from model.mlp import MLP
from train.eval import evaluate


def get_scores(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_bundle = DatasetBundle.load(paths.DATASET_BUNDLE_DIR)
    aggr_route_lookup = load_route_lookup(paths.DATASETS_DIR + cfg.dataset.route_aggr)

    model = MLP(cfg.model.input_dim, cfg.model.mlp.hidden_dims, cfg.model.output_dim)
    model.load_state_dict(torch.load("/home3/s3799174/bus-travel-time-prediction/outputs/2025-07-21/19-17-36/MLP.pth"))
    model.to(device)
    train_loader, val_loader, test_loader = create_dataloaders(cfg, dataset_bundle, aggr_route_lookup,
                                                               aggr_collate_fn, num_workers=4)
    mae, _, _, id_targets = evaluate(cfg, model, val_loader, device)
    metadata = pd.read_parquet(paths.DATASETS_DIR + "dataset_metadata.parquet")
    merged_df = id_targets.merge(metadata, on="id")
    print(merged_df.shape)
    print(merged_df.head)
    merged_df.to_parquet(paths.RESULTS_DIR + "result_analysis.parquet")


@hydra.main(config_path=paths.CONFIG_DIR, config_name="config", version_base=None)
def heatmap(cfg: Config):
    get_scores(cfg)
    # id_targets = np.load(paths.RESULTS_DIR + "MLP_dataset_time_id_targets.npy")
    # print(id_targets.shape)
    # # metadata = pd.read_parquet(paths.DATASETS_DIR + "dataset_metadata.parquet")
    #
    # preds_df = pd.DataFrame(id_targets, columns=["id", "prediction"])
    # preds_df["id"] = preds_df["id"].astype(int)
    #
    # merged_df = metadata.merge(preds_df, on="id")
    #
    # print(merged_df.shape)
    # print(merged_df.head)


if __name__ == '__main__':
    heatmap()