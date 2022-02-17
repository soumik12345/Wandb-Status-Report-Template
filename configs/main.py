import ml_collections
from fastai.vision.all import *


def get_wandb_configs() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.project = "CamVid"
    config.entity = "av-demo"
    config.job_type = "sweep"
    config.artifact_id = "camvid-dataset:v0"

    return config


def get_experiment_configs() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.seed = 123
    config.batch_size = 8
    config.image_height = 720
    config.image_width = 960
    config.image_resize_factor = 4
    config.validation_split = 0.2
    config.backbone = "mobilenetv2_100"
    config.hidden_dims = 256
    config.num_epochs = 5
    config.loss_function = "categorical_cross_entropy"
    config.learning_rate = 1e-3
    config.fit = "fit"

    return config


def get_loss_mappings() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.categorical_cross_entropy = (CrossEntropyLossFlat,)
    config.focal = (FocalLossFlat,)
    config.dice = DiceLoss

    return config


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.experiment_configs = get_experiment_configs()
    config.wandb_configs = get_wandb_configs()
    config.loss_mappings = get_loss_mappings()
    config.sweep_count = 5

    return config
