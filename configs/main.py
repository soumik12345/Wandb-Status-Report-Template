import ml_collections

import wandb_configs, experiment


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    
    config.experiment_configs = experiment.get_config()
    config.wandb_configs = wandb_configs.get_config()

    return config
