import click
from src.utils import get_project_root


@click.group()
def features():
    """Process data into features."""
    pass


@click.command()
@click.argument("filepaths", nargs=-1)
@click.pass_context
def run(ctx, filepaths):
    """Run feature generation pipeline (data/raw -> data/interim)

    Parameters
    ----------
    ctx : click.Context
        Click current context object
    filepaths : list of str
        List of filepaths
    """
    output_dir = str(get_project_root(plus="data/interim"))


features.add_command(run)
