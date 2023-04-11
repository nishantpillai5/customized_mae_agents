import random
import time

import click

from src.utils import get_project_root


@click.group()
def world():
    """Build world"""
    pass


@click.command()
@click.pass_context
def run(ctx):
    """Run world.

    Parameters
    ----------
    ctx : click.Context
        Click current context object
    filepaths : list of str
        List of filepaths
    """

    from src.world import world_utils

    cycles = 200
    env = world_utils.env(render_mode="human", max_cycles=cycles)

    env.reset()
    current_cycle = 0
    agent_count = 4  # TODO: get from env
    action_queue = []

    for agent in env.agent_iter():
        if current_cycle >= cycles * agent_count:
            break
        if current_cycle % agent_count == 0:
            adversary_0_action = random.choice([0, 1, 2, 3, 4])
            adversary_1_action = random.choice([0, 1, 2, 3, 4])
            adversary_2_action = random.choice([0, 1, 2, 3, 4])
            good_agent_action = random.choice([0, 1, 2, 3, 4])

            action_queue += [
                adversary_0_action,
                adversary_1_action,
                adversary_2_action,
                good_agent_action,
            ]
        # print(agent)
        env.render()
        # obs, reward, done, info = env.last()
        observation, cumulative_rewards, terminations, truncations, infos = env.last()
        action = action_queue.pop(0)
        env.step(action)
        current_cycle += 1

        # Following this but it's not working: https://github.com/openai/multiagent-particle-envs/issues/76
        # score+=reward
    else:
        env.close()


world.add_command(run)
