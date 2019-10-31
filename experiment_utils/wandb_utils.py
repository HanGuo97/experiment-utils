import wandb
import click
from typing import Dict, Optional


def wandb_initialize(
        project_name: str,
        experiment_name: str,
        experiment_tag: Optional[str]=None,
        experiment_notes: Optional[str]=None) -> None:
    
    if not isinstance(project_name, str):
        raise TypeError("`project_name` must be String")

    if not isinstance(experiment_name, str):
        raise TypeError("`experiment_name` must be String")

    click.echo(click.style(f"Initializing W&B Project "
                           f"{project_name}: "
                           f"{experiment_name}",
                           fg="green"))

    wandb.init(project=project_name,
               name=experiment_name,
               tags=[experiment_tag],
               notes=experiment_notes,
               sync_tensorboard=True)


def wandb_log(d: Dict, **kwargs) -> None:
    return wandb.log(d, **kwargs)
