import click


@click.group()
def visualization():
    """Visualize"""
    pass


@click.command()
@click.argument("filepath")
@click.pass_context
def run(ctx, filepath):
    """Run visualization pipeline

    Parameters
    ----------
    ctx : click.Context
        Click current context object
    filepath : str
        filepath to visualize
    """
    # ctx.invoke(visualize, filepath=filepath)


visualization.add_command(run)
