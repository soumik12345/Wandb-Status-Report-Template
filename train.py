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


# FLAGS = flags.FLAGS
# config_flags.DEFINE_config_file("experiment_configs")


def train_fn(configs: ml_collections.ConfigDict):
    wandb_configs = configs.wandb_configs
    experiment_configs = configs.experiment_configs
    loss_alias_mappings = configs.loss_mappings
    inference_config = configs.inference

    run = wandb.init(
        name = wandb_configs.name,
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
    
    # store model checkpoints and JIT
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
    
    ## Inference benchmark
    model_file = f"Unet_{wandb.config.backbone}_traced.pt"
    torch.cuda.empty_cache()
    inference_time = benchmark_inference_time(model_file,
                        batch_size=inference_config.batch_size,
                        image_shape=(wandb.config.image_height,
                                     wandb.config.image_width),
                        num_warmup_iters=inference_config.warmup,
                        num_iter=inference_config.num_iter,
                        resize_factor=inference_config.resize_factor,
                        seed=wandb.config.seed
                        )
    wandb.log({"inference_time":inference_time})

    
from configs import get_config
if __name__=="__main__":
    cfg = get_config()
    train_fn(cfg)