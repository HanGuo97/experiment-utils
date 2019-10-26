import os
import click
import stringcase
from git import Repo
from absl import logging
from datetime import datetime
from collections import namedtuple
from typing import Tuple, Optional
logging.set_verbosity(logging.INFO)

ExperimentConfig = namedtuple(
    "ExperimentConfig", (
        "project_name",
        "experiment_tag",
        "experiment_name",
        "experiment_description",
        "logdir",
        "repo"))

CONFIG: Optional[ExperimentConfig] = None


def interactive_initialize(
        base_log_dir: str=".",
        default_project_name: Optional[str]=None,
        default_experiment_tag: Optional[str]=None,
        default_experiment_description: Optional[str]=None
) -> ExperimentConfig:
    
    if not isinstance(base_log_dir, str):
        raise ValueError("`base_log_dir` must be String")

    # Set file-scope configuration
    global CONFIG

    # Query user inputs
    project_name = click.prompt(
        "Please Enter The Project Name",
        type=str, default=default_project_name)

    experiment_tag = click.prompt(
        "Please Enter The Experiment Name",
        type=str, default=default_experiment_tag)

    experiment_description = click.prompt(
        "Please Enter The Experiment Description",
        type=str, default=default_experiment_description)

    # Tag should be in camel-case
    experiment_tag = stringcase.camelcase(experiment_tag)
    # Experiment Name will include more info
    experiment_name, git_repo = _get_experiment_name(tag=experiment_tag)
    # Experiment log dir
    logdir = os.path.join(base_log_dir, experiment_name)

    CONFIG = ExperimentConfig(
        repo=git_repo,
        logdir=logdir,
        project_name=project_name,
        experiment_tag=experiment_tag,
        experiment_name=experiment_name,
        experiment_description=experiment_description)

    _print_config()
    click.confirm("Do you want to continue?", abort=True)

    return CONFIG


def _get_experiment_name(tag: Optional[str]=None) -> Tuple[str, Repo]:
    """Get the experiment name based on Git status and time"""
    repo = Repo("./")
    date = datetime.now()

    name = (
        f"{date.year}{date.month}{date.day}_"
        f"BRANCH_{repo.active_branch}_"
        f"COMMIT_{repo.head.commit.hexsha[:5]}_"
        f"TAG_{tag}")

    return name, repo


def _print_config() -> None:
    if CONFIG is None:
        return
    # Maximum field lengths
    max_length = max(map(len, CONFIG._fields))

    for i in range(len(CONFIG)):
        logging.info(f"{CONFIG._fields[i]: <{max_length}}: {CONFIG[i]}")
