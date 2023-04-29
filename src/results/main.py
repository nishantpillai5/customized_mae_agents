import click

from src.agent.player import STRAT


@click.group()
def results():
    """Results"""
    pass


@click.command()
@click.argument("adversary_model")
@click.option(
    "--eps-num",
    "-e",
    default=6,
    help="Episodes to test",
)
@click.option(
    "--max-cycles",
    "-m",
    default=1000,
    help="Max cycles",
)
@click.pass_context
def eval(ctx, adversary_model, eps_num, max_cycles):
    import glob
    import logging
    import logging.config
    import os
    import random
    from pprint import pformat

    import numpy as np
    import torch
    from gymnasium.utils.save_video import save_video

    from src.agent.constants import AGENTS, cfg, device
    from src.agent.utils import DQN, StateCache, select_action
    from src.utils import get_logging_conf

    def env_creator(render_mode="rgb_array"):
        from src.world import world_utils

        env = world_utils.env(render_mode=render_mode, max_cycles=max_cycles)
        return env

    worker_config = get_logging_conf(f"eval")
    logging.config.dictConfig(worker_config)
    logger = logging.getLogger("test")

    reward_dict = {s: {s: None for s in cfg["strats"]} for s in cfg["strats"]}

    model_dict = {}

    for s in cfg["strats"]:
        for m in glob.glob(adversary_model + "*policy.pth"):
            if s in m:
                model_dict.update({s: m})

    print("Models:")
    for key, value in model_dict.items():
        print(value)

    for model_strategy in cfg["strats"]:
        # FIXME: Get number of actions from gym action space
        n_actions = 5

        env = env_creator(render_mode="rgb_array")
        env.reset()
        env.render()

        state, _, _, _, _ = env.last()
        # Get the number of state observations
        n_observations = len(state)

        policy_net = DQN(n_observations, n_actions, cfg).to(device)
        # target_net = DQN(n_observations, n_actions, cfg).to(device)
        policy_net.load_state_dict(torch.load(model_dict[model_strategy]))
        # target_net.load_state_dict(torch.load(PATH))
        policy_net.eval()

        for player_strategy in cfg["strats"]:
            steps_done = 0
            episode_durations = []
            episode_rewards = []

            state_cache = StateCache()

            for i_episode in range(eps_num):
                env.reset()
                env.render()
                state, reward, _, _, _ = env.last()

                for agent in AGENTS:
                    state_cache.save_state(agent, state, reward)

                rewards = []
                actions = {
                    agent: torch.tensor([[0]], device=device) for agent in AGENTS
                }
                t = 0
                for agent in env.agent_iter():
                    t += 1
                    # agent = AGENTS[t % 4]
                    previous_state = state_cache.get_state(agent)
                    observation, reward, terminated, truncated, _ = env.last()

                    rewards.append(reward)

                    observation, reward = state_cache.deal_state(
                        agent, observation, reward
                    )
                    old_action = actions[agent]
                    done = terminated or truncated
                    if done:
                        env.step(None)
                    else:
                        action, steps_done = select_action(
                            cfg,
                            observation,
                            policy_net,
                            good_agent=("agent" in agent),
                            steps_done=steps_done,
                            random_action=env.action_space("agent_0").sample(),
                            player_strat=player_strategy,
                        )
                        actions[agent] = action
                        env.step(action.item())

                    env.render()

                    if done:  # FIXME: is there a reason for two checks lol?
                        episode_durations.append(t + 1)
                        episode_rewards += rewards[-4:]
                        logger.info(f"m{model_strategy[0]}p{player_strategy[0]} Ep reward: {episode_rewards[-4:]}")
                        break

            logger.info("m{model_strategy[0]}p{player_strategy[0]} Complete Ep reward: \n" + pformat(np.asarray(episode_rewards)))
            reward_dict[model_strategy][player_strategy] = (
                np.asarray(episode_rewards).reshape((eps_num, 4)).sum(axis=1)
            )

    stats = {}
    for model_strat in cfg["strats"]:
        for s1_player_strat in cfg["strats"]:
            for s2_player_strat in cfg["strats"]:
                sample1 = reward_dict[model_strat][s1_player_strat]
                sample2 = reward_dict[model_strat][s2_player_strat]
                statistic, pvalue = scipy.stats.ttest_ind(
                    sample1, sample2, equal_var=False  # FIXME: Not sure
                )
                stats.update(
                    {
                        f"m{model_strat[0]}p{s1_player_strat[0]}_vs_m{model_strat[0]}p{s2_player_strat[0]}": (statistic, pvalue)
                    }
                )
    logger.info("T-test: \n" + pformat(stats))


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
    "--eps-num",
    "-e",
    default=3,
    help="Episodes to record",
)
@click.option(
    "--max-cycles",
    "-m",
    default=1000,
    help="Max cycles",
)
@click.pass_context
def record(ctx, adversary_model, strategy, eps_num, max_cycles):
    import logging
    import logging.config
    import os
    import random
    from pprint import pformat

    import numpy as np
    import torch
    from gymnasium.utils.save_video import save_video

    from src.agent.constants import AGENTS, cfg, device
    from src.agent.utils import DQN, StateCache, select_action
    from src.utils import get_logging_conf

    def env_creator(render_mode="rgb_array"):
        from src.world import world_utils

        env = world_utils.env(render_mode=render_mode, max_cycles=max_cycles)
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

    policy_net = DQN(n_observations, n_actions, cfg).to(device)
    # target_net = DQN(n_observations, n_actions, cfg).to(device)
    policy_net.load_state_dict(torch.load(adversary_model))
    # target_net.load_state_dict(torch.load(PATH))
    policy_net.eval()

    steps_done = 0
    episode_durations = []
    episode_rewards = []

    state_cache = StateCache()

    for i_episode in range(eps_num):
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
                    cfg,
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
                episode_durations.append(t + 1)
                episode_rewards += rewards[-4:]
                logger.info(f"Ep reward: {episode_rewards[-4:]}")
                save_video(
                    frames,
                    "logs/videos",
                    # fps=env.metadata["render_fps"]*4, # Sped up
                    fps=60,
                    name_prefix=os.path.basename(adversary_model)
                    + f"_{strategy}_"
                    + str(np.around(np.sum(episode_rewards[-4:]), decimals=3)),
                )
                break

    # TODO: Aggregate and Log Rewards
    logger.info("Complete Ep reward: \n" + pformat(np.asarray(episode_rewards)))


results.add_command(record)
results.add_command(eval)

