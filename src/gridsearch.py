import hydra

from config import paths
from config.config import Config


@hydra.main(config_path=paths.CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: Config):
    gs_dropout = [0.0, 0.1, 0.2]
    gs_hidden_dims = [[32], [64, 32], [128, 64, 32]]
    gs_learning_rate = [1e-3, 5e-3, 1e-4]
    weight_decay = [0.0, 1e-5, 1e-4]
    optimizer = ["Adam", "AdamW"]

    print(cfg.training.batch_size)
    cfg.training.batch_size = 12
    print(cfg.training.batch_size)


if __name__ == "__main__":
    main()