import ml_collections

from .wandb_configs import get_config as get_wandb_configs
from .experiment import get_config as get_experiment_configs
from .loss import get_config as get_loss_mappings
from .sweep import get_config as get_sweep_configs


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.experiment_configs = get_experiment_configs()
    config.wandb_configs = get_wandb_configs()
    config.loss_mappings = get_loss_mappings()
    config.sweep_configs = get_sweep_configs()
    config.sweep_count = 5

    return config
