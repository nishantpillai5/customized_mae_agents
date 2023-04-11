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


pipeline.add_command(run)
