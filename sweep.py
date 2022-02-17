import wandb
from functools import partial
from fastai.vision.all import *

from absl import app
from absl import flags

import ml_collections
from ml_collections.config_flags import config_flags

from segmentation.camvid_utils import *
from segmentation.train_utils import *
from segmentation.metrics import *


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("sweep_configs")


SWEEP_CONFIGS = {
    "method": "bayes",
    "metric": {"name": "foreground_acc", "goal": "maximize"},
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 5,
    },
    "parameters": {
        "batch_size": {"values": [4, 8, 16]},
        "image_resize_factor": {"values": [2, 4]},
        "backbone": {
            "values": [
                "mobilenetv2_100",
                "mobilenetv3_small_050",
                "mobilenetv3_large_100",
                "resnet18",
                "resnet34",
                "resnet50",
                "vgg19",
                "vgg16",
            ]
        },
        "loss_function": {"values": ["categorical_cross_entropy", "focal", "dice"]},
        "learning_rate": {"values": [1e-2, 1e-3, 1e-4]},
        "fit": {"values": ["fit", "fine-tune"]},
    },
}


def train_fn(configs: ml_collections.ConfigDict):
    wandb_configs = configs.wandb_configs
    experiment_configs = configs.experiment_configs
    loss_alias_mappings = configs.loss_mappings

    wandb.init(
        project=wandb_configs.project,
        entity=wandb_configs.entity,
        job_type=wandb_configs.job_type,
        config=experiment_configs.to_dict(),
    )

    data_loader, class_labels = get_dataloader(
        artifact_id=configs.wandb_configs.artifact_id,
        batch_size=wandb.config.batch_size,
        image_shape=(wandb.config.image_height, wandb.config.image_width),
        resize_factor=wandb.config.image_resize_factor,
        validation_split=wandb.config.validation_split,
        seed=wandb.config.seed,
    )

    learner = get_learner(
        data_loader,
        backbone=wandb.config.backbone,
        hidden_dim=wandb.config.hidden_dims,
        num_classes=len(class_labels),
        checkpoint_file=None,
        loss_func=loss_alias_mappings[wandb.config.loss_function](axis=1),
        metrics=[DiceMulti(), foreground_acc],
        log_preds=False,
    )

    if wandb.config.fit == "fit":
        learner.fit_one_cycle(wandb.config.num_epochs, wandb.config.learning_rate)
    else:
        learner.fine_tune(wandb.config.num_epochs, wandb.config.learning_rate)

    wandb.log(
        {f"Predictions_Table": table_from_dl(learner, learner.dls.valid, class_labels)}
    )

    save_model_to_artifacts(
        learner.model,
        f"Unet_{wandb.config.backbone}",
        image_shape=(wandb.config.image_height, wandb.config.image_width),
        artifact_name=f"{run.name}-saved-model",
        metadata={
            "backbone": wandb.config.backbone,
            "hidden_dims": wandb.config.hidden_dims,
            "input_size": (wandb.config.image_height, wandb.config.image_width),
            "class_labels": class_labels,
        },
    )


def main(_):
    config = FLAGS.sweep_configs
    sweep_id = wandb.sweep(
        SWEEP_CONFIGS,
        project=config.wandb_configs.project,
        entity=config.wandb_configs.entity,
    )
    wandb.agent(
        sweep_id, function=partial(train_fn, config), count=config.sweep_count
    )


if __name__ == "__main__":
    app.run(main)
