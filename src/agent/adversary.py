import logging
import logging.config
from pathlib import Path
from pprint import pformat

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

    from src.agent.constants import MAX_CYCLES, device
    from src.agent.utils import DQN

    def env_creator(render_mode="rgb_array"):
        from src.world import world_utils

        env = world_utils.env(render_mode=render_mode, max_cycles=MAX_CYCLES)
        return env

    env = env_creator(render_mode="human" if visualize else "rgb_array")
    env.reset()
    env.render()

    state, _, _, _, _ = env.last()
    n_observations = len(state)

    policy_net = DQN(n_observations, n_actions).to(device)

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
    default="",
    help="Description of the run",
)
@click.pass_context
def train(ctx, visualize, desc):
    import random
    from itertools import count

    import torch
    import torch.optim as optim
    import wandb

    from src.agent.constants import (
        AGENTS,
        EPS_DECAY,
        EPS_END,
        EPS_NUM,
        EPS_START,
        GAMMA,
        LR,
        MAX_CYCLES,
        RAY_BATCHES,
        BATCH_SIZE,
        REPLAY_MEM,
        TAU,
        TEST_EPS_NUM,
        TEST_MAX_CYCLES,
        TEST_RAY_BATCHES,
        device,
    )
    from src.agent.utils import (
        DQN,
        ReplayMemory,
        StateCache,
        optimize_model,
        print_rewards,
        select_action,
    )

    print("Running with", RAY_BATCHES, "baches")
    print("Running for", EPS_NUM, "episodes")
    print("An episode is", MAX_CYCLES, "cycles")

    def env_creator(render_mode="rgb_array"):
        from src.world import world_utils

        env = world_utils.env(render_mode=render_mode, max_cycles=MAX_CYCLES)
        return env

    import ray

    PLAYER_STRAT = "evasive"

    @ray.remote
    def ray_train(name=None):
        if not name:
            name = int(random.random() * 10000)

        worker_config = get_logging_conf(f"ad_train_{name}")
        logging.config.dictConfig(worker_config)
        logger = logging.getLogger("both")

        filename = worker_config["handlers"]["r_file"]["filename"]

        wandb_run = wandb.init(
            project="customized_mae_agents",
            entity="ju-ai-thesis",
            config={
                "description": desc,
                "filename": filename,
                "strategy": PLAYER_STRAT,
                # Hyperparameters
                "eps_num": EPS_NUM,
                "learning_rate": LR,
                "max_cycles": MAX_CYCLES,
                "ray_batches": RAY_BATCHES,
                "batch_size": BATCH_SIZE,
                "replay": REPLAY_MEM,
                "tau": TAU,
                "gamma": GAMMA,
                "eps_start": EPS_START,
                "eps_end": EPS_END,
                "eps_decay": EPS_DECAY,
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

        policy_net = DQN(n_observations, n_actions).to(device)
        target_net = DQN(n_observations, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())

        optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
        memory = ReplayMemory(REPLAY_MEM)
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
                        player_strat=PLAYER_STRAT,
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
                        optimize_model(optimizer, memory, policy_net, target_net)
                        target_net_state_dict = target_net.state_dict()
                        policy_net_state_dict = policy_net.state_dict()
                        for key in policy_net_state_dict:
                            target_net_state_dict[key] = policy_net_state_dict[
                                key
                            ] * TAU + target_net_state_dict[key] * (1 - TAU)
                        target_net.load_state_dict(target_net_state_dict)
                env.render()

                if done:  # FIXME: is there a reason for two checks lol?
                    episode_durations.append(t + 1)
                    episode_rewards += rewards[-4:]
                    rewards = np.array(rewards)
                    log_data = {
                        "episode_rewards": episode_rewards[-4:],
                        "avg_ep_reward": np.sum(rewards) / (MAX_CYCLES * 3),
                        "num_collisions": (rewards > 0).sum() // (3),
                        "distance_penalty": (
                            np.sum(rewards[(rewards < 0)]) / (MAX_CYCLES * 3)
                        ),
                    }
                    logger.info(f"Ep reward: {log_data['episode_rewards']}")
                    logger.info(f"This ep avg reward: {log_data['avg_ep_reward']}")
                    logger.info(f"Num of collisions: {log_data['num_collisions']}")
                    wandb.log(log_data)

                    break

        logger.info("Complete Ep reward: \n" + pformat(np.asarray(episode_rewards)))
        # Save model
        torch.save(policy_net.state_dict(), filename + "_policy.pth")
        torch.save(target_net.state_dict(), filename + "_target.pth")

        artifact = wandb.Artifact(
            name=filename[filename.index("/") + 1 :], type="model"
        )
        artifact.add_file(local_path=filename + "_policy.pth")
        wandb_run.log_artifact(artifact)

        return torch.tensor(episode_rewards, dtype=torch.float)

    task_handles = []
    try:
        for i in range(1, RAY_BATCHES + 1):
            task_handles.append(ray_train.remote(name=i))

        output = ray.get(task_handles)
        print(output)
    except KeyboardInterrupt:
        for i in task_handles:
            ray.cancel(i)


adversary.add_command(train)
adversary.add_command(eval)
