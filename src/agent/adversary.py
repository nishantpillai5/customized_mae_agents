import logging
import logging.config
from pathlib import Path
from pprint import pformat, pprint

import click
import numpy as np

from src.utils import get_files, get_logging_conf, get_project_root


@click.group()
def adversary():
    """Adversary processes"""
    pass


@click.command()
@click.argument("filepaths", nargs=-1)
@click.option(
    "--visualize",
    "-v",
    is_flag=True,
    show_default=True,
    default=False,
    help="Visualize",
)
@click.pass_context
def eval(ctx, filepaths, visualize):
    import torch

    from src.agent.constants import cfg, device
    from src.agent.utils import DQN

    def env_creator(render_mode="rgb_array"):
        from src.world import world_utils

        env = world_utils.env(render_mode=render_mode, max_cycles=cfg["max_cycles"])
        return env

    env = env_creator(render_mode="human" if visualize else "rgb_array")
    env.reset()
    env.render()

    state, _, _, _, _ = env.last()
    n_observations = len(state)

    policy_net = DQN(n_observations, n_actions, cfg).to(device)

    # https://pytorch.org/tutorials/beginner/saving_loading_models.html

    for filepath in filepaths:  # to test all models with one command
        policy_net.load_state_dict(torch.load(filepath))

        # you must call model.eval() (maybe not needed)
        # https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html#save-and-load-the-model-via-state-dict

        policy_net.eval()
        print("Eval: ", filepath)

        # TODO: compute rewards with saved models and log them, make a csv, maybe use ray for multiple instances


@click.command()
@click.option(
    "--visualize",
    "-v",
    is_flag=True,
    show_default=True,
    default=False,
    help="Visualize",
)
@click.option(
    "--desc",
    "-d",
    default=None,
)
@click.pass_context
def train(ctx, visualize, desc):
    import random
    from itertools import count

    import torch
    import torch.optim as optim

    import wandb
    from src.agent.constants import AGENTS, cfg, device
    from src.agent.utils import (
        DQN,
        ReplayMemory,
        StateCache,
        optimize_model,
        select_action,
    )
    from src.results.main import record

    print(f"Running for {','.join(cfg['strats'])} strats")
    print(f"Running for {cfg['eps_num']} episodes")
    print(f"An episode is {cfg['max_cycles']} cycles")

    def env_creator(render_mode="rgb_array"):
        from src.world import world_utils

        env = world_utils.env(render_mode=render_mode, max_cycles=cfg["max_cycles"])
        return env

    import ray

    @ray.remote
    def ray_train(player_strat, name=None):
        if not name:
            name = int(random.random() * 10000)

        worker_config = get_logging_conf("ad_train", suffix=player_strat)
        logging.config.dictConfig(worker_config)
        logger = logging.getLogger("both")

        filename = worker_config["handlers"]["r_file"]["filename"]

        wandb_run = wandb.init(
            name=("" if desc is None else desc) + " " + player_strat,
            project="customized_mae_agents",
            entity="ju-ai-thesis",
            config={
                "description": desc,
                "filename": filename,
                "strategy": player_strat,
                # Hyperparameters
                **cfg,
            },
        )
        # FIXME: Get number of actions from gym action space
        n_actions = 5

        env = env_creator(render_mode="human" if visualize else "rgb_array")
        env.reset()
        env.render()

        state, _, _, _, _ = env.last()
        # Get the number of state observations
        n_observations = len(state)

        policy_net = DQN(n_observations, n_actions, cfg).to(device)
        target_net = DQN(n_observations, n_actions, cfg).to(device)
        target_net.load_state_dict(policy_net.state_dict())

        optimizer = optim.AdamW(
            policy_net.parameters(), lr=cfg["learning_rate"], amsgrad=True
        )
        memory = ReplayMemory(cfg["replay_mem"])
        steps_done = 0

        episode_durations = []
        episode_rewards = []

        state_cache = StateCache()

        for i_episode in range(cfg["eps_num"]):
            if player_strat != "multiple":
                player_agent_strat = player_strat
            else:
                player_agent_strat = ["evasive", "hiding", "shifty"][i_episode % 3]
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
                        cfg,
                        observation,
                        policy_net,
                        good_agent=("agent" in agent),
                        steps_done=steps_done,
                        random_action=env.action_space("agent_0").sample(),
                        player_strat=player_agent_strat,
                    )
                    actions[agent] = action
                    env.step(action.item())

                    if "agent" not in agent:  # if adversary
                        memory.push(
                            previous_state[0],
                            old_action,
                            observation,
                            previous_state[1],
                        )
                        optimize_model(
                            optimizer, memory, policy_net, target_net, cfg=cfg
                        )
                        target_net_state_dict = target_net.state_dict()
                        policy_net_state_dict = policy_net.state_dict()
                        for key in policy_net_state_dict:
                            target_net_state_dict[key] = policy_net_state_dict[
                                key
                            ] * cfg["tau"] + target_net_state_dict[key] * (
                                1 - cfg["tau"]
                            )
                        target_net.load_state_dict(target_net_state_dict)
                env.render()

                if done:  # FIXME: is there a reason for two checks lol?
                    episode_durations.append(t + 1)
                    episode_rewards += rewards[-4:]
                    rewards = np.array(rewards)
                    log_data = {
                        "episode_rewards": episode_rewards[-4:],
                        "avg_ep_reward": np.sum(rewards) / (cfg["max_cycles"] * 3),
                        "num_collisions": (rewards > 0).sum() // (3),
                        "distance_penalty": (
                            np.sum(rewards[(rewards < 0)]) / (cfg["max_cycles"] * 3)
                        ),
                    }
                    logger.info(f"Ep reward: {log_data['episode_rewards']}")
                    logger.info(f"This ep avg reward: {log_data['avg_ep_reward']}")
                    logger.info(f"Num of collisions: {log_data['num_collisions']}")
                    wandb.log(log_data)

                    break

        logger.info("Complete Ep reward: \n" + pformat(np.asarray(episode_rewards)))
        # Save model
        model_filename = filename + "_policy.pth"
        torch.save(policy_net.state_dict(), model_filename)
        torch.save(target_net.state_dict(), filename + "_target.pth")

        artifact = wandb.Artifact(
            name=filename[filename.index("/") + 1 :], type="model"
        )
        artifact.add_file(local_path=model_filename)
        wandb_run.log_artifact(artifact)

        # Record
        # ctx.invoke(
        #     record,
        #     adversary_model=model_filename,
        #     strategy=player_agent_strat,
        #     eps_num=3,
        #     max_cycles=cfg["max_cycles"],
        # )

        return torch.tensor(episode_rewards, dtype=torch.float)

    ray_batches = cfg["strats"]*cfg["ray_batches"]
    task_handles = []
    try:
        for i in range(len(ray_batches)):
            task_handles.append(ray_train.remote(ray_batches[i], name=i))

        output = ray.get(task_handles)
        print(output)
    except KeyboardInterrupt:
        for i in task_handles:
            ray.cancel(i)


@click.command()
@click.option(
    "--num-samples",
    "-s",
    default=1000,
    help="No of samples",
)
@click.pass_context
def tune(ctx, num_samples):
    import os
    import random

    import ray
    import torch
    import torch.optim as optim
    from ray import tune
    from ray.air import session

    from src.agent.constants import AGENTS, define_search_space, device, hpo_cfg
    from src.agent.utils import (
        DQN,
        ReplayMemory,
        StateCache,
        optimize_model,
        select_action,
    )

    static_config = hpo_cfg

    PLAYER_STRAT = "multiple"

    worker_config = get_logging_conf(f"ad_tune")
    logging.config.dictConfig(worker_config)
    logger = logging.getLogger("both")
    filename = worker_config["handlers"]["r_file"]["filename"]

    def trainable(config):
        name = None
        visualize = False

        def env_creator(render_mode="rgb_array"):
            from src.world import world_utils

            env = world_utils.env(
                render_mode=render_mode, max_cycles=static_config["max_cycles"]
            )
            return env

        if not name:
            name = int(random.random() * 10000)

        # FIXME: Get number of actions from gym action space
        n_actions = 5

        env = env_creator(render_mode="human" if visualize else "rgb_array")
        env.reset()
        env.render()

        state, _, _, _, _ = env.last()
        # Get the number of state observations
        n_observations = len(state)

        policy_net = DQN(n_observations, n_actions, config).to(device)
        target_net = DQN(n_observations, n_actions, config).to(device)
        target_net.load_state_dict(policy_net.state_dict())

        optimizer = optim.AdamW(
            policy_net.parameters(), lr=config["learning_rate"], amsgrad=True
        )
        memory = ReplayMemory(config["replay_mem"])
        steps_done = 0

        episode_durations = []
        episode_rewards = []

        state_cache = StateCache()
        ep_avg_rewards = []

        for i_episode in range(static_config["eps_num"]):
            if PLAYER_STRAT != "multiple":
                player_agent_strat = PLAYER_STRAT
            else:
                player_agent_strat = ["evasive", "hiding", "shifty"][i_episode % 3]
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
                        config,
                        observation,
                        policy_net,
                        good_agent=("agent" in agent),
                        steps_done=steps_done,
                        random_action=env.action_space("agent_0").sample(),
                        player_strat=player_agent_strat,
                    )
                    actions[agent] = action
                    env.step(action.item())

                    if "agent" not in agent:  # if adversary
                        memory.push(
                            previous_state[0],
                            old_action,
                            observation,
                            previous_state[1],
                        )
                        optimize_model(
                            optimizer,
                            memory,
                            policy_net,
                            target_net,
                            cfg={
                                "batch_size": config["batch_size"],
                                "gamma": config["gamma"],
                            },
                        )
                        target_net_state_dict = target_net.state_dict()
                        policy_net_state_dict = policy_net.state_dict()
                        for key in policy_net_state_dict:
                            target_net_state_dict[key] = policy_net_state_dict[
                                key
                            ] * config["tau"] + target_net_state_dict[key] * (
                                1 - config["tau"]
                            )
                        target_net.load_state_dict(target_net_state_dict)
                env.render()

                if done:  # FIXME: is there a reason for two checks lol?
                    episode_durations.append(t + 1)
                    episode_rewards += rewards[-4:]
                    rewards = np.array(rewards)
                    ep_avg_rewards.append(
                        np.sum(rewards) / (static_config["max_cycles"] * 3)
                    )
                    log_data = {
                        "episode_rewards": episode_rewards[-4:],
                        "avg_ep_reward": np.sum(rewards)
                        / (static_config["max_cycles"] * 3),
                        "num_collisions": (rewards > 0).sum() // (3),
                        "distance_penalty": (
                            np.sum(rewards[(rewards < 0)])
                            / (static_config["max_cycles"] * 3)
                        ),
                    }
                    break
                    # Intermediate scoring
                    # session.report({"avg_ep_avg_reward": np.sum(ep_avg_rewards) / len(ep_avg_rewards)})
        # Final scoring
        # return {"avg_ep_avg_reward": np.sum(ep_avg_rewards) / len(ep_avg_rewards)}
        session.report(
            {"avg_ep_avg_reward": np.sum(ep_avg_rewards) / len(ep_avg_rewards)}
        )

    # config = {x: tune.uniform(y[0], y[1]) for x, y in search_space_cfg.items()}

    from optuna.samplers import NSGAIISampler
    from ray.tune.search.optuna import OptunaSearch

    algo = OptunaSearch(
        define_search_space,
        metric="avg_ep_avg_reward",
        mode="max",
        sampler=NSGAIISampler(),
    )

    # space = {x: (y[0], y[1]) for x, y in search_space_cfg.items()}

    # from ray.tune.search.bayesopt import BayesOptSearch

    # algo = BayesOptSearch(
    #     space,
    #     metric="avg_ep_avg_reward",
    #     mode="max",
    #     # utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0},
    # )

    tuner = tune.Tuner(
        # tune.with_resources(
        #     tune.with_parameters(trainable), resources={"gpu": 1}
        # ),
        trainable,
        # param_space=config,
        tune_config=tune.TuneConfig(
            metric="avg_ep_avg_reward",
            mode="max",
            search_alg=algo,
            num_samples=num_samples,
        ),
        # run_config=ray.air.RunConfig(stop={"training_iteration": 20}),
    )
    results = tuner.fit()
    best_result = results.get_best_result()  # Get best result object

    logger.info("Log: " + str(best_result.log_dir))
    logger.info(
        "Best config: \n" + pformat(best_result.config)
    )  # Get best trial's hyperparameters
    logger.info(
        "Best metrics: \n" + pformat(best_result.metrics)
    )  # Get best trial's last results
    results.get_dataframe().to_csv(
        f"logs/{os.path.basename(filename)}.csv", index=False
    )


adversary.add_command(train)
adversary.add_command(tune)
adversary.add_command(eval)
