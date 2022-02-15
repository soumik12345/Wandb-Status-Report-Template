import wandb
import torch
from fastai.vision.all import *
from segmentation.camvid_utils import *
from segmentation.train_utils import *


PROJECT = "CamVid"
ENTITY = "av-demo"
ARTIFACT_ID = "av-demo/CamVid/camvid-dataset:v0"
JOB_TYPE = "sweep"


EXPERIMENT_CONFIG = {
    "seed": 123,
    "batch_size": 8,
    "image_height": 720,
    "image_width": 960,
    "image_resize_factor": 4,
    "validation_split": 0.2,
    "backbone": "mobilenetv2_100",
    "hidden_dims": 256,
    "num_epochs": 5,
    "loss_function": "categorical_cross_entropy",
    "learning_rate": 1e-3,
    "fit": "fit",
}


LOSS_ALIAS_MAPPING = {
    "categorical_cross_entropy": CrossEntropyLossFlat,
    "focal": FocalLossFlat,
    "dice": DiceLoss,
}


SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "foreground_acc", "goal": "maximize"},
    "early_terminate": {"type": "hyperband", "min_iter": 5,},
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


def train_fn():
    run = wandb.init(
        project=PROJECT, entity=ENTITY, job_type=JOB_TYPE, config=EXPERIMENT_CONFIG
    )

    data_loader, class_labels = get_dataloader(
        artifact_id=ARTIFACT_ID,
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
        loss_func=LOSS_ALIAS_MAPPING[wandb.config.loss_function](axis=1),
        metrics=[DiceMulti(), foreground_acc],
        log_preds=False,
    )

    if wandb.config.fit == "fit":
        learner.fit_one_cycle(wandb.config.num_epochs, wandb.config.learning_rate)
    else:
        learner.fine_tune(wandb.config.num_epochs, wandb.config.learning_rate)

    samples, outputs, predictions = get_predictions(learner)
    table = create_wandb_table(samples, outputs, predictions, class_labels)
    wandb.log({f"Baseline_Predictions_{run.name}": table})

    model = learner.model.eval()
    torch.cuda.empty_cache()
    del learner
    wandb.log({"Model_Parameters": get_model_parameters(model)})
    wandb.log(
        {
            "Inference_Time": benchmark_inference_time(
                model,
                image_shape=(wandb.config.image_height, wandb.config.image_width),
                batch_size=8,
                num_iter=20,
                seed=wandb.config.seed,
            )
        }
    )


if __name__ == "__main__":
    sweep_id = wandb.sweep(SWEEP_CONFIG, project=PROJECT)
    wandb.agent(sweep_id, function=train_fn, count=5)
