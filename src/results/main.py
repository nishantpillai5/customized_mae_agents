import click

from src.agent.player import STRAT


@click.group()
def results():
    """Results"""
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
@click.pass_context
def record(ctx, adversary_model, strategy):
    import logging
    import logging.config
    import random
    from pprint import pformat

    import numpy as np
    import torch
    from gymnasium.utils.save_video import save_video

    from src.agent.constants import (
        ACTIONS,
        AGENTS,
        EPS_NUM,
        MAX_CYCLES,
        RAY_BATCHES,
        device,
    )
    from src.agent.utils import DQN, StateCache, select_action
    from src.utils import get_logging_conf

    def env_creator(render_mode="rgb_array"):
        from src.world import world_utils

        env = world_utils.env(render_mode=render_mode, max_cycles=MAX_CYCLES)
        return env

    worker_config = get_logging_conf(f"record")
    logging.config.dictConfig(worker_config)
    logger = logging.getLogger("test")

    # FIXME: Get number of actions from gym action space
    n_actions = 5

    env = env_creator(render_mode="rgb_array")
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
        frames = []
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

            frames.append(env.render())

            if done:  # FIXME: is there a reason for two checks lol?
                save_video(
                    frames,
                    "logs/videos",
                    fps=env.metadata["render_fps"],
                    step_starting_index=0,
                    episode_index=i_episode,
                )
                episode_durations.append(t + 1)
                episode_rewards += rewards[-4:]
                logger.info(f"Ep reward: {episode_rewards[-4:]}")
                break

    # TODO: Aggregate and Log Rewards
    logger.info("Complete Ep reward: \n" + pformat(np.asarray(episode_rewards)))


results.add_command(record)
