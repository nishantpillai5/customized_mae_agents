import logging
import logging.config
from pathlib import Path

import click

from src.utils import get_files, get_logging_conf, get_project_root

logging.config.dictConfig(get_logging_conf("pipeline"))
logger = logging.getLogger("both")


@click.group()
def pipeline():
    """Process raw data into processed data"""
    pass


@click.command()
@click.argument("filepaths", nargs=-1)
@click.pass_context
def run(ctx, filepaths):
    """Run entire pipeline

    Parameters
    ----------
    ctx : click.Context
        Click current context object
    filepaths : list of str
        List of filepaths
    """

    raw_dir = str(get_project_root(plus="data/raw"))
    interim_dir = str(get_project_root(plus="data/interim"))
    processed_dir = str(get_project_root(plus="data/processed"))

    for filepath in filepaths:
        filename = Path(filepath).stem

        # Data
        ctx.invoke(data_run, filepaths=[filepath], output_dir=raw_dir)

        # Features
        data_output_files = get_files(raw_dir + f"/{filename}/*.npy")
        ctx.invoke(features_run, filepaths=data_output_files, output_dir=interim_dir)

        # Predict
        features_output_files = get_files(interim_dir + f"/{filename}/*_result.npy")

        ctx.invoke(
            models_run,
            filepaths=features_output_files,
            output_dir=processed_dir,
        )


@click.command()
# @click.argument("filepaths", nargs=-1)
@click.pass_context
def wandb_test(ctx):
    import random

    import wandb

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="customized_mae_agents",
        entity="ju-ai-thesis",
        # track hyperparameters and run metadata
        config={
            "learning_rate": 0.02,
            "architecture": "CNN",
            "dataset": "CIFAR-100",
            "epochs": 10,
        },
    )

    # simulate training
    epochs = 10
    offset = random.random() / 5
    for epoch in range(2, epochs):
        acc = 1 - 2**-epoch - random.random() / epoch - offset
        loss = 2**-epoch + random.random() / epoch + offset

        # log metrics to wandb
        wandb.log({"acc": acc, "loss": loss})

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()


pipeline.add_command(run)
pipeline.add_command(wandb_test)
