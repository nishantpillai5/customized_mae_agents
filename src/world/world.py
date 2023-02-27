import time
import random

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

    from pettingzoo.mpe import simple_tag_v2

    env = simple_tag_v2.env(render_mode="human")
    env.reset()

    for _ in range(200):
        env.render()
        # time.sleep(1/20)
        # n_state, reward, done, info = env.step(action)
        for agent in env.agent_iter():
            action = random.choice([0,1,2,3,4])
            #print(agent)
            env.render()
            # obs, reward, done, info = env.last()
            observation, cumulative_rewards, terminations, truncations, infos = env.last()
            env.step(3)

        # Following this but it's not working: https://github.com/openai/multiagent-particle-envs/issues/76
        # score+=reward
    else:
        env.close()


world.add_command(run)
