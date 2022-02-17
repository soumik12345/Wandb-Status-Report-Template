import ml_collections


def get_config() -> ml_collections.ConfigDict:
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
