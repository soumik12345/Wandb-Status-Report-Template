import ml_collections


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.project = "CamVid"
    config.entity = "av-demo"
    config.job_type = "sweep"
    config.artifact_id = "camvid-dataset:v0"

    return config
