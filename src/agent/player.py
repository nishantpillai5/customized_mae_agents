import logging
import logging.config
from pathlib import Path

import click
from src.utils import get_files, get_logging_conf, get_project_root

from pprint import pprint

logging.config.dictConfig(get_logging_conf("player"))
logger = logging.getLogger("test")


def away_from_everything(state):  # Evasive
    """
    Player A: Away from everything
    This player's intention is to get away from the enemies or from everythin as quick as possible.

    Getting the vector:
    - Get enemies and obstacles positions
    - For each element
        - Get opposite direction
        - Multiply by a factor of the current distance to that element
        - Possibly multiply by a factor based on whether the element is an enemy or an obstacle (higher for enemies)
    - Get the sum of the found vectors

    Given the vector, at every frame:
    - Move in that direction at fixed speed

    The algorithm for getting the vector can be run every frame or every n frames.

    Possible design flaws:
    - If the map is infinite, this player might always win because enemies are slower
    - If the map is finite, the player might need the notion of the existance of walls

    How to beat this player:
    - Enemies have to learn to surround the player and slowly move together towards it
    """
    return None


def separate_from_enemies(state):  # Hiding
    """
    Player B: Separate from enemies
    This player's intention is to use the group of obstacles as separation from the enemies, so that the enemies are forced to circumnavigate the obstacles.

    Getting the vector:
    - Get enemies and obstacles positions
    - Get the average obstacle position (AOP) (this could be a manually set position)
    - Convert enemies and self positions as radial positions to the AOP
    - For each enemy
        - Get opposite tangential direction (clockwise or counterclockwise)
        - Get opposite radial direction (away or towards AOP)
        - Multiply them by distance to that enemy
    - Get the sum of the found vectors


    Given the vector, at every frame:
    - Move in that direction at fixed speed

    Possible design flaws:
    - If the map is infinite, this player might always win because enemies are slower
    - If the map is finite, the player might need the notion of the existance of walls

    How to beat this player:
    - Enemies have to learn to put themselves at various distances to the AOP, so to impede player movement, and then get closer.
    """
    return None


def using_obstacles(state):  # Shifty
    """
    Player C: using obstacles to get away
    This player chooses a desired position on the map based on the situation, and then pathfinds towards it

    A set of manually selected positions is given to the player to choose from based on the situation.
    The positions are manually selected like so:
        - Indentify the spaces between close obstacles
        - Put one position at the center of the spaces, and two other positions before and after, to create a tunnel
        - Put other positions at the sides of the map

    Selecting the target position:
    - Get the 3 closest positions to the player and the 2 more distant ones
    - For each of them, select the one that's more distant to the enemies (probably by looking at the closest enemy to that)
    - If all of them are close to enemies, repeat for all positions rather than just these 5

    Given the position, at every frame:
    - Use a PathFinding algorithm to get the movement
    - PathFinding should include obstacles as obstacles, but also interpret enemies as obstacles with a certain range

    Possible design flaws:
    - We don't currently have a Pathfinding solution lol

    How to beat this player:
    - Enemies should find a particular set of obstacles to guard from a distance, to induce the player into it
    - They should also learn to put one of themselves close to the other possible player target positions
    """
    return None


def anti_gravity(state):
    """
    """
    return None


def dqn(state):
    """
    Player D: AI vs AI test
    This is a replica of the originally intended test for this environment, applied to our settings.
    """
    return None


def dummy(state):
    """
    Getting the vector:
    - Get closest player or obstacle
    - Find direction to get away from them

    At every frame:
    - Follow vector`
    """
    return None


STRAT = {
    "away_from_everything": away_from_everything,
    "separate_from_enemies": separate_from_enemies,
    "using_obstacles": using_obstacles,
    "anti_gravity": anti_gravity,
    "dqn": dqn,
    "dummy": dummy,
}


def get_player_action(state, strategy=None, override=None):
    # TODO: Player strategies
    if override is not None:  # FIXME: Temporary override with random action
        return override
    if strategy is not None:
        return STRAT[strategy](state)
    else:
        return STRAT["dummy"](state)


@click.group()
def player():
    """Player processes"""
    pass


@click.command()
@click.argument("adversary_model")
@click.option("--visualize", "-v", is_flag=True, show_default=True, default=False, help="Visualize")
@click.pass_context
def test(ctx, adversary_model, visualize):
    # adversary_model -> filename
    # Load model and run player strat on that model
    pass


player.add_command(test)
