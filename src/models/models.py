import click
from src.utils import get_project_root


@click.group()
def models():
    """Train model or run inference"""
    pass


@click.command()
@click.argument("filepaths", nargs=-1)
@click.pass_context
def run(ctx, filepaths):
    """Run inference pipeline (data/interim -> data/processed)

    Parameters
    ----------
    ctx : click.Context
        Click current context object
    filepaths : list of str
        List of filepaths
    """
    output_dir = str(get_project_root(plus="data/processed"))


models.add_command(run)
