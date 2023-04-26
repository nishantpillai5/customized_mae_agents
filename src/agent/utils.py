import math
import random
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.agent.constants import device
from src.agent.player import get_player_action

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class StateCache:
    def __init__(self):
        self.value = {}

    def save_state(self, agent, ns, reward):
        self.value[agent] = {
            "obs": torch.tensor(ns, dtype=torch.float32, device=device).unsqueeze(0),
            "reward": torch.tensor([reward], device=device),
        }

    def get_state(self, agent):
        return self.value[agent]["obs"], self.value[agent]["reward"]

    def deal_state(self, agent, ns, reward):
        self.save_state(agent, ns, reward)
        return self.get_state(agent)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, cfg):
        super(DQN, self).__init__()

        layers = cfg["num_layers"] * [cfg["num_neurons"]]
        layers = [n_observations, *layers, n_actions]

        self.layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )

        # self.layer1 = nn.Linear(n_observations, 64)
        # self.layer2 = nn.Linear(64, 32)
        # self.layer3 = nn.Linear(32, 20)
        # self.layer4 = nn.Linear(20, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # x = F.relu(self.layer1(x))
        # x = F.relu(self.layer2(x))
        # x = F.relu(self.layer3(x))
        # return self.layer4(x)

        for layer in self.layers:
            x = layer(x)

        return x


def optimize_model(optimizer, memory, policy_net, target_net, cfg):
    if len(memory) < cfg["batch_size"]:
        return
    transitions = memory.sample(cfg["batch_size"])
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(cfg["batch_size"], device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * cfg["gamma"]) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def plot_durations(show_result=False):
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if torch.max(durations_t) == durations_t[-1]:
        print(name, "::", durations_t[-1])


def print_rewards(name, episode_rewards):
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if torch.max(rewards_t) in rewards_t[-4:]:
        print("rew:", name, rewards_t[-4:])
    else:
        print("rew2:", name, rewards_t[-4:])


def select_action(
    cfg,
    state,
    policy_net,
    good_agent=False,
    steps_done=0,
    random_action=None,
    player_strat=None,
):
    override = random_action if player_strat is None else None
    player_action = get_player_action(state, strategy=player_strat, override=override)

    if good_agent:
        if player_action is None:
            player_action = get_player_action(
                state, strategy=None, override=random_action
            )
        return (
            torch.tensor([[player_action]], device=device, dtype=torch.long),
            steps_done,
        )
    else:
        sample = random.random()
        eps_threshold = cfg["eps_end"] + (cfg["eps_start"] - cfg["eps_end"]) * math.exp(
            -1.0 * steps_done / cfg["eps_decay"]
        )
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return policy_net(state).max(1)[1].view(1, 1), steps_done
        else:
            return (
                torch.tensor([[random_action]], device=device, dtype=torch.long),
                steps_done,
            )
