import os
import random
import pandas as pd
from glob import glob
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb
import fastai
from fastai.vision.all import *
from fastai.callback.wandb import WandbCallback


path = untar_data(URLs.CAMVID)
codes = np.loadtxt(path/'codes.txt', dtype=str)
fnames = get_image_files(path/"images")
class_labels = {k: v for k, v in enumerate(codes)}


def label_func(fn):
    return path/"labels"/f"{fn.stem}_P{fn.suffix}"


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        groups: int = 1,
        stride: int = 1,
        activation: bool = True
    ):
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=groups,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        ] + (
            [nn.ReLU6(inplace=True)] if activation else []
        )
        super().__init__(*layers)

        
class CRPBlock(nn.Module):
    """A bunch of convs and a maxpool with a tricky forward"""
    def __init__(self, in_channels, out_channels, num_stages=1, use_groups=False):
        super().__init__()
        groups = in_channels if use_groups else 1
        convs = [nn.Conv2d(
            in_channels if (i == 0) else out_channels,
            out_channels, kernel_size=1, bias=False, groups=groups
        ) for i in range(num_stages)]
        self.convs = nn.ModuleList(convs)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        "y = x + f(x) + f(f(x)) + f(f(f(x)))..."
        out = x
        for conv in self.convs:
            out = conv(self.pool(out))
            x = out + x
        return x

    
class UnetBlock(nn.Module):

    def __init__(
        self, in_up, in_side, out_channels, kernel_size=1, num_stages=4, use_groups=False
    ):
        super().__init__()
        self.conv_up   = ConvLayer(in_up, out_channels, kernel_size)
        self.conv_side = ConvLayer(in_side, out_channels, kernel_size)
        self.crp = CRPBlock(
            out_channels, out_channels, num_stages=num_stages, use_groups=use_groups
        )

    def forward(self, side_input, up_input):
        up_input = self.conv_up(up_input)
        side_input = self.conv_side(side_input)
        if up_input.shape[-2:] != side_input.shape[-2:]:
            up_input = F.interpolate(
                up_input, size=side_input.shape[-2:], mode="bilinear", align_corners=False
            )
        out = self.crp(F.relu(up_input + side_input))
        return out

    
class DynamicUnet(nn.Module):
    """
    A Unet that take almost any backbone from timm
    Reference: https://github.com/tcapelle/hydra_net/blob/master/hydranet/models.py#L13
    """
    def __init__(self, backbone="mobilenetv2_100", dim=256):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=True, features_only=True)
        # passing dummy tensor to get sizes
        dummy_tensor = torch.rand([1, 3, 64, 64])
        features = self.encoder(dummy_tensor)
        ch_sizes = [list(f.shape)[1] for f in features][::-1]
        self.upsample_blocks = nn.ModuleList()
        self.mid_conv = ConvLayer(ch_sizes[0], dim, 3)
        for i, ch_size in enumerate(ch_sizes[1:]):
            self.upsample_blocks.append(
                UnetBlock(
                    dim, ch_size, out_channels=dim, use_groups=(i==(len(features)-2))
                )
            )

    def forward(self, x):
        input_shape = x.shape
        # features reversed in order
        features = self.encoder(x)[::-1]
        # put last feature on dim of the model
        x = self.mid_conv(features[0])
        # upsample blocks with shortcurts from the sides
        for f, ublock in zip(features[1:], self.upsample_blocks):
            x = ublock(f, x)
        x = F.interpolate(x, size=input_shape[-2:], mode="bilinear", align_corners=False)
        return x

    
class SegmentationModel(nn.Module):
    def __init__(self, backbone="mobilenetv2_100", hidden_dim=256, num_classes=21):
        super().__init__()
        self.backbone = DynamicUnet(backbone, dim=hidden_dim)
        self.segmentation_head = nn.Sequential(
            ConvLayer(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1, bias=False)
        )

    def forward(self, x):
        backbone_out = self.backbone(x)
        return self.segmentation_head(backbone_out)

    
def _one_hot(x, classes, axis=1):
    "Target mask to one hot"
    return torch.stack([torch.where(x==c, 1,0) for c in range(classes)], axis=axis)

class DiceLoss:
    "Dice coefficient metric for binary target in segmentation"
    def __init__(self, axis=1, smooth=1): 
        store_attr()
    
    def __call__(self, pred, targ):
        targ = _one_hot(targ, pred.shape[1])
        pred, targ = flatten_check(self.activation(pred), targ)
        inter = (pred * targ).sum()
        union = (pred + targ).sum()
        return 1 - (2. * inter + self.smooth) / (union + self.smooth)
    
    def activation(self, x):
        return F.softmax(x, dim=self.axis)
    
    def decodes(self, x):
        return x.argmax(dim=self.axis)

    
def get_predictions(learner):
    inputs, predictions, targets, outputs = learner.get_preds(with_input=True, with_decoded=True)
    x, y, samples, outputs = learner.dls.valid.show_results(
        tuplify(inputs) + tuplify(targets), outputs, show=False, max_n=36
    )
    return samples, outputs, predictions


def create_wandb_table(samples, outputs, predictions):
    "Creates a wandb table with predictions and targets side by side"
    table = wandb.Table(columns=["Image", "Predicted Mask", "Ground Truth"])
    for (image, label), pred_label in zip(samples, outputs):
        image = image.permute(1,2,0)
        table.add_data(
            wandb.Image(image),
            wandb.Image(
                image,
                masks={
                    "predictions":  {
                        'mask_data':  pred_label[0].numpy(),
                        'class_labels':class_labels
                    }
                }
            ),
            wandb.Image(
                image,
                masks={
                    "ground truths": {
                        'mask_data': label.numpy(),
                        'class_labels':class_labels
                    }
                }
            )
        )
    return table


experiment_configs = {
    "batch_size": 8,
    "image_height": 720,
    "image_width": 960,
    "image_resize_factor": 4,
    "backbone": "mobilenetv2_100",
    "num_epochs": 5,
    "loss_function": "categorical_cross_entropy",
    "learning_rate": 1e-3,
    "fit": "fit"
}


def train_fn():
    run = wandb.init(
        project="autonomous-vehicle-status-report",
        entity="geekyrakshit",
        config=experiment_configs
    )
    data_loader = SegmentationDataLoaders.from_label_func(
        path, bs=wandb.config.batch_size,
        fnames=fnames, label_func=label_func, codes=codes, 
        item_tfms=Resize((
            wandb.config.image_height // wandb.config.image_resize_factor,
            wandb.config.image_width // wandb.config.image_resize_factor
        )),
    )
    segmentation_model = SegmentationModel(num_classes=len(codes))
    loss_fn = None
    if wandb.config.loss_function == "categorical_cross_entropy":
        loss_fn = CrossEntropyLossFlat(axis=1)
    elif wandb.config.loss_function == "focal":
        loss_fn = FocalLossFlat(axis=1)
    elif wandb.config.loss_function == "dice":
        loss_fn = DiceLoss(axis=1)
    learner = Learner(
        data_loader,
        segmentation_model,
        loss_func=loss_fn,
        metrics=[DiceMulti(), foreground_acc],
        cbs=[
            SaveModelCallback(fname=f"unet_{wandb.config.backbone}"),
            MixedPrecision(),
            WandbCallback(log_preds=False)
        ]
    )
    if wandb.config.fit == "fit":
        learner.fit_one_cycle(
            wandb.config.num_epochs,
            wandb.config.learning_rate
        )
    else:
        learner.fine_tune(
            wandb.config.num_epochs,
            wandb.config.learning_rate
        )
    samples, outputs, predictions = get_predictions(learner)
    table = create_wandb_table(samples, outputs, predictions)
    wandb.log({f"Predictions-{run.name}": table})
    wandb.log({"Model Params": sum(p.numel() for p in segmentation_model.parameters())})

    
sweep_config_backbone = {
    "method": "bayes",
    "metric": {"name": "valid_loss", "goal": "minimize"},
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
                "vgg16"
            ]
        },
        "learning_rate": {"values": [1e-2, 1e-3, 1e-4]},
        "fit": {"values": ["fit", "fine-tune"]}
    },
}


sweep_id = wandb.sweep(sweep_config_backbone, project="autonomous-vehicle-status-report")
wandb.agent(sweep_id, function=train_fn)
