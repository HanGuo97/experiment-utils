import wandb
from typing import Dict
from absl import logging


def wandb_initialize(
        project_name: str,
        experiment_name: str) -> None:
    
    if not isinstance(project_name, str):
        raise TypeError("`project_name` must be String")

    if not isinstance(experiment_name, str):
        raise TypeError("`experiment_name` must be String")

    logging.info(f"Initializing W&B Project "
                 f"{project_name}: "
                 f"{experiment_name}")
    
    wandb.init(project=project_name,
               name=experiment_name,
               sync_tensorboard=True)


def wandb_log(d: Dict, **kwargs) -> None:
    return wandb.log(d, **kwargs)
