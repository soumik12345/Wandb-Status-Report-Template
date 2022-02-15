import wandb
from typing import Tuple
from fastai.vision.all import *


def label_func(fn):
    return fn.parent.parent / "labels" / f"{fn.stem}_P{fn.suffix}"


def get_dataloader(
    artifact_id: str,
    batch_size: int,
    image_shape: Tuple[int, int],
    resize_factor: int,
    validation_split: float,
    seed: int,
):
    """Grab an artifact and creating a Pytorch DataLoader"""
    artifact = wandb.use_artifact(artifact_id, type="dataset")
    artifact_dir = Path(artifact.download())
    codes = np.loadtxt(artifact_dir / "codes.txt", dtype=str)
    fnames = get_image_files(artifact_dir / "images")
    class_labels = {k: v for k, v in enumerate(codes)}
    return (
        SegmentationDataLoaders.from_label_func(
            artifact_dir,
            bs=batch_size,
            fnames=fnames,
            label_func=label_func,
            codes=codes,
            item_tfms=Resize(
                (image_shape[0] // resize_factor, image_shape[1] // resize_factor)
            ),
            valid_pct=validation_split,
            seed=seed,
        ),
        class_labels,
    )
