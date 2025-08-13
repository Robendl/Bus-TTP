import hydra

from config import paths
from config.config import Config


@hydra.main(config_path=paths.CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: Config):
    print(cfg.training.batch_size)
    cfg.training.batch_size = 12
    print(cfg.training.batch_size)


if __name__ == "__main__":
    main()