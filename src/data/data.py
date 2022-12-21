import click
from src.utils import get_project_root


@click.group()
def data():
    """Process raw data into processed data."""
    pass


@click.command()
@click.argument("filepaths", nargs=-1)
@click.pass_context
def run(ctx, filepaths):
    """Run data processing pipeline (data/raw -> data/raw)

    Parameters
    ----------
    ctx : click.Context
        Click current context object
    filepaths : list of str
        List of filepaths
    """
    output_dir = str(get_project_root(plus="data/raw"))


data.add_command(run)
