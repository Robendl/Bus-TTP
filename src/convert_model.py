"""Export the trained LSTM model to ONNX for downstream deployment."""
import hydra
import torch

import config.paths as paths
from config.config import Config
from data.build_dataset import load_route_lookup
from data.data_processing import create_dataloaders
from data.dataset_bundle import DatasetBundle
from model.lstm import LSTMFeedforwardCombination
from runtime import setup_environment

setup_environment()


@hydra.main(config_path=paths.CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: Config):
    dataset_bundle = DatasetBundle.load(paths.DATASET_BUNDLE_DIR, cfg)
    seq_route_lookup = load_route_lookup(cfg, paths.DATASETS_DIR + cfg.dataset.route_seq)

    lstm_input_dim = next(iter(seq_route_lookup.values())).shape[1]
    ff_input_dim = dataset_bundle.train.x.shape[1] - 3

    model = LSTMFeedforwardCombination(cfg, lstm_input_dim, ff_input_dim)
    model.load_state_dict(torch.load(cfg.eval.checkpoint_path, map_location="cpu"))
    model.eval()

    train_loader, *_ = create_dataloaders(
        cfg, dataset_bundle, seq_route_lookup, is_route_sequence=True, num_workers=0
    )
    _, (time_features, padded_routes, lengths), _ = next(iter(train_loader))
    dummy_input = (time_features[:1], padded_routes[:1], lengths[:1])

    output_path = f"{paths.RESULTS_DIR}/onnx/LSTM.onnx"
    print(f"Exporting LSTM to {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["time_features", "padded_routes", "lengths"],
        output_names=["output"],
        dynamic_axes={
            "time_features": {0: "batch_size"},
            "padded_routes": {0: "batch_size", 1: "seq_len"},
            "lengths": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )


if __name__ == "__main__":
    main()
