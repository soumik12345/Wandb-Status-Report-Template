import ml_collections
from fastai.vision.all import *


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    
    config.categorical_cross_entropy = CrossEntropyLossFlat,
    config.focal = FocalLossFlat,
    config.dice = DiceLoss

    return config
