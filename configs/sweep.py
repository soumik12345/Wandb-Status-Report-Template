sweep_configs = {
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
