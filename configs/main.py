import ml_collections

from .wandb_configs import get_config as get_wandb_configs
from .experiment import get_config as get_experiment_configs


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    
    config.experiment_configs = get_experiment_configs()
    config.wandb_configs = get_wandb_configs()

    return config
