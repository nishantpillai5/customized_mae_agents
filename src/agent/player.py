import random

import click

from src.agent.constants import ACTIONS


'''
state vector elements:
 0, 1: self.velocity
 2, 3: self.position
 4-21: landmarks relative positions 
22-27: adversaries relative positions
28-33: adversaries velocities
'''


def evasive_player(state):
    """
    Player A: Away from everything
    This player's intention is to get away from the enemies or from everything as quick as possible.

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

    # Getting sum of relative positions
    x_diffs = 0
    y_diffs = 0
    # for landmarks
    for i in range(4, 22, 2):
        x_diffs += state[0][i].item() * 0.2
    for i in range(5, 23, 2):
        y_diffs += state[0][i].item() * 0.2
    # for adversaries    
    for i in range(22, 28, 2):
        x_diffs += state[0][i].item()
    for i in range(23, 29, 2):
        y_diffs += state[0][i].item()
    
    # Checking larger difference and moving in opposite direction
    action = ACTIONS["no_action"]
    if abs(x_diffs) > abs(y_diffs):
        if x_diffs > 0:
            action = ACTIONS["move_left"]
        elif x_diffs < 0:
            action = ACTIONS["move_right"]
    elif abs(x_diffs) < abs(y_diffs):
        if y_diffs > 0:
            action = ACTIONS["move_down"]
        elif y_diffs < 0:
            action = ACTIONS["move_up"]



    return action


def hiding_player(state):
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


def shifty_player(state):
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


def dqn_player(state):
    """
    Player D: AI vs AI test
    This is a replica of the originally intended test for this environment, applied to our settings.
    """
    return None


def dummy_player(state):
    """
    Getting the vector:
    - Get closest player or obstacle
    - Find direction to get away from them

    At every frame:
    - Follow vector
    """
    return None


def static_player(state):
    """
    Doesn't move.
    """
    return 0


def random_player(state):
    return random.choice(list(ACTIONS.values()))


STRAT = {
    "evasive": evasive_player,
    "hiding": hiding_player,
    "shifty": shifty_player,
    "dqn": dqn_player,
    "dummy": dummy_player,
    "static": static_player,
    "random": random_player,
}


def get_player_action(state, strategy=None, override=None):
    if override is not None:
        action = override
    if strategy is not None:
        action = STRAT[strategy](state)
    else:
        action = STRAT["static"](state)

    # boundary check
    if (
        state[0][2].item() < -1
        and action == ACTIONS["move_left"]
        or state[0][2].item() > 1
        and action == ACTIONS["move_right"]
        or state[0][3].item() < -1
        and action == ACTIONS["move_down"]
        or state[0][3].item() > 1
        and action == ACTIONS["move_up"]
    ):
        action = ACTIONS["no_action"]
    return action


@click.group()
def player():
    """Player processes"""
    pass


@click.command()
@click.argument("adversary_model")
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(list(STRAT.keys())),
    default="static",
    help="Strategy",
)
@click.option(
    "--visualize",
    "-v",
    is_flag=True,
    show_default=True,
    default=False,
    help="Visualize",
)
@click.pass_context
def test(ctx, adversary_model, strategy, visualize):
    import logging
    import logging.config
    import random
    from pprint import pformat

    import numpy as np
    import torch

    from src.agent.constants import AGENTS, EPS_NUM, MAX_CYCLES, RAY_BATCHES, device
    from src.agent.utils import DQN, StateCache, select_action
    from src.utils import get_logging_conf

    def env_creator(render_mode="rgb_array"):
        from src.world import world_utils

        env = world_utils.env(render_mode=render_mode, max_cycles=MAX_CYCLES)
        return env

    import ray

    @ray.remote
    def ray_test(name=None):
        if not name:
            name = int(random.random() * 10000)

        worker_config = get_logging_conf(f"player_{name}")
        logging.config.dictConfig(worker_config)
        logger = logging.getLogger("both")

        # FIXME: Get number of actions from gym action space
        n_actions = 5

        env = env_creator(render_mode="human" if visualize else "rgb_array")
        env.reset()
        env.render()

        state, _, _, _, _ = env.last()
        # Get the number of state observations
        n_observations = len(state)

        policy_net = DQN(n_observations, n_actions).to(device)
        # target_net = DQN(n_observations, n_actions).to(device)
        policy_net.load_state_dict(torch.load(adversary_model))
        # target_net.load_state_dict(torch.load(PATH))
        policy_net.eval()

        steps_done = 0
        episode_durations = []
        episode_rewards = []

        state_cache = StateCache()

        for i_episode in range(EPS_NUM):
            env.reset()
            env.render()
            state, reward, _, _, _ = env.last()

            for agent in AGENTS:
                state_cache.save_state(agent, state, reward)

            rewards = []
            actions = {agent: torch.tensor([[0]], device=device) for agent in AGENTS}
            t = 0
            for agent in env.agent_iter():
                t += 1
                # agent = AGENTS[t % 4]
                previous_state = state_cache.get_state(agent)
                observation, reward, terminated, truncated, _ = env.last()

                rewards.append(reward)

                observation, reward = state_cache.deal_state(agent, observation, reward)
                old_action = actions[agent]
                done = terminated or truncated
                if done:
                    env.step(None)
                else:
                    action, steps_done = select_action(
                        observation,
                        policy_net,
                        good_agent=("agent" in agent),
                        steps_done=steps_done,
                        random_action=env.action_space("agent_0").sample(),
                        player_strat=strategy,
                    )
                    actions[agent] = action
                    env.step(action.item())

                env.render()

                if done:  # FIXME: is there a reason for two checks lol?
                    episode_durations.append(t + 1)
                    episode_rewards += rewards[-4:]
                    logger.info(f"Ep reward: {episode_rewards[-4:]}")
                    break

        # TODO: Aggregate and Log Rewards
        logger.info("Complete Ep reward: \n" + pformat(np.asarray(episode_rewards)))

        return torch.tensor(episode_rewards, dtype=torch.float)

    task_handles = []
    try:
        for i in range(1, RAY_BATCHES + 1):
            task_handles.append(ray_test.remote(name=i))

        output = ray.get(task_handles)
        print(output)
    except KeyboardInterrupt:
        for i in task_handles:
            ray.cancel(i)


player.add_command(test)
