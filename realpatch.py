from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from run import Config, DatasetConfig, MatchingConfig, start

# ============== register config classes ================

# ConfigStore enables type validation
cs = ConfigStore.instance()
# Register schemas
cs.store(name="main_schema", node=Config)
# Data
cs.store(name="celeba_schema", node=DatasetConfig, package="data")
# Matching
cs.store(name="matching_realpatch_schema", node=MatchingConfig, package="matching")


# Look for and load config file named "config.yaml" in "conf" folder
@hydra.main(config_path='conf', config_name='matching_config')
def runner(cfg: Config) -> None:
    OmegaConf.update(cfg, "results_dir", Path().resolve(), merge=False)
    start(cfg=cfg)


if __name__ == "__main__":
    runner()
