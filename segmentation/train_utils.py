import wandb
import torch
from tqdm import tqdm
from typing import Tuple, List, Dict

from fastai.vision.all import *
from fastai.callback.wandb import WandbCallback

from .camvid_utils import get_dataloader
from .model import SegmentationModel


def get_model_parameters(model):
    with torch.no_grad():
        num_params = sum(p.numel() for p in model.parameters())
    return num_params


def get_predictions(learner):
    inputs, predictions, targets, outputs = learner.get_preds(
        with_input=True, with_decoded=True
    )
    x, y, samples, outputs = learner.dls.valid.show_results(
        tuplify(inputs) + tuplify(targets), outputs, show=False, max_n=36
    )
    return samples, outputs, predictions


def benchmark_inference_time(
    model,
    image_shape: Tuple[int, int],
    batch_size: int,
    num_warmup_iters: int,
    num_iter: int,
    seed: int,
):
    data_loader, _ = get_dataloader(
        artifact_id="av-demo/CamVid/camvid-dataset:v0",
        batch_size=batch_size,
        image_shape=image_shape,
        resize_factor=2,
        validation_split=0.2,
        seed=seed,
    )

    dummy_input = torch.randn(
        batch_size, 3, image_shape[0] // 2, image_shape[0] // 2, dtype=torch.float
    ).to("cuda")

    starter, ender = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    timings = np.zeros((num_iter, 1))

    print("Warming up GPU...")
    for _ in tqdm(range(num_warmup_iters)):
        _ = model(dummy_input)

    print(
        f"Computing inference time over {num_iter} iterations with batches of {batch_size} images..."
    )

    with torch.no_grad():
        for step in tqdm(range(num_iter)):
            x, y = next(iter(data_loader.valid))
            starter.record()
            _ = model(x)
            ender.record()
            torch.cuda.synchronize()
            timings[step] = starter.elapsed_time(ender)

    return np.sum(timings) / (num_iter * batch_size)


def create_wandb_table(samples, outputs, class_labels):
    "Creates a wandb table with predictions and targets side by side"
    table = wandb.Table(columns=["Image", "Predicted Mask", "Ground Truth"])
    for (image, label), pred_label in zip(samples, outputs):
        image = image.permute(1, 2, 0)
        table.add_data(
            wandb.Image(image),
            wandb.Image(
                image,
                masks={
                    "predictions": {
                        "mask_data": pred_label[0].numpy(),
                        "class_labels": class_labels,
                    }
                },
            ),
            wandb.Image(
                image,
                masks={
                    "ground truths": {
                        "mask_data": label.numpy(),
                        "class_labels": class_labels,
                    }
                },
            ),
        )
    return table


def get_learner(
    data_loader,
    backbone: str,
    hidden_dim: int,
    num_classes: int,
    checkpoint_file: Union[None, str, Path],
    loss_func,
    metrics: List,
    log_preds: bool = False,
):
    model = SegmentationModel(backbone, hidden_dim, num_classes=num_classes)
    save_model_callback = SaveModelCallback(fname=f"unet_{backbone}")
    mixed_precision_callback = MixedPrecision()
    wandb_callback = WandbCallback(log_preds=log_preds)
    learner = Learner(
        data_loader,
        model,
        loss_func=loss_func,
        metrics=metrics,
        cbs=[save_model_callback, mixed_precision_callback, wandb_callback],
    )
    if checkpoint_file is not None:
        learner.load(checkpoint_file)
    return learner


def save_model_to_artifacts(
    model,
    model_name: str,
    image_shape: Tuple[int, int],
    artifact_name: str,
    metadata: Dict,
):
    print("Saving model using scripting...")
    saved_model_script = torch.jit.script(model)
    saved_model_script.save(model_name + "_script.pt")
    example_forward_input = torch.randn(
        1, 3, image_shape[0] // 2, image_shape[0] // 2, dtype=torch.float
    ).to("cuda")
    print("Saving model using tracing...")
    saved_model_traced = torch.jit.trace(model, example_inputs=example_forward_input)
    saved_model_traced.save(model_name + "_traced.pt")
    artifact = wandb.Artifact(artifact_name, type="model", metadata=metadata)
    artifact.add_file(saved_model_script)
    artifact.add_file(saved_model_traced)
    wandb.log_artifact(artifact)
