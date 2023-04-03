import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

cycles = 200
def env_creator(render_mode="rgb_array", cycles=200):
    from src.world import world_utils
    env = world_utils.env(render_mode=render_mode, max_cycles=cycles)
    return env


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

#plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


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

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

#plt.ioff()
#plt.show()
steps_done = 0 # for some reason functions have it as global, so I have to put it here outside lol


import ray
@ray.remote
def do_the_flop(name=None):
    if not name:
        name = int(random.random()*10000)
    # Get number of actions from gym action space
    n_actions = 5
    # Get the number of state observations

    env = env_creator(render_mode="human")
    env.reset()
    env.render()
    state, _, _, _, _ =env.last()
    n_observations = len(state)

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(256) # 00)  # rob: reduced to reduce RAM usage


    steps_done = 0

    def select_action(state, good_agent=False):
        global steps_done
        if good_agent:
            return torch.tensor([[env.action_space("agent_0").sample()]], device=device, dtype=torch.long)
            #pass # player strategies
        else:
            sample = random.random()
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    # t.max(1) will return the largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    return policy_net(state).max(1)[1].view(1, 1)
            else:
                return torch.tensor([[env.action_space("agent_0").sample()]], device=device, dtype=torch.long)


    episode_durations = []


    def plot_durations(show_result=False):
        durations_t = torch.tensor(episode_durations, dtype=torch.float)
        if torch.max(durations_t) == durations_t[-1]:
            print(name, "::", durations_t[-1])

    episode_rewards = []

    def print_rewards():
        rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
        if torch.max(rewards_t) in rewards_t[-4:]:
            print(name, "::", rewards_t[-4:])

    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
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
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

    if torch.cuda.is_available():
        num_episodes = 600
    else:
        num_episodes = 50

    state_cache = {}
    def save_state(agent, ns, reward):
        state_cache[agent] = {"obs": torch.tensor(ns, dtype=torch.float32, device=device).unsqueeze(0),
                              "reward": torch.tensor([reward], device=device)}
    def get_state(agent):
        return state_cache[agent]["obs"], state_cache[agent]["reward"]
    def deal_state(agent, ns, reward):
        save_state(agent, ns, reward)
        return get_state(agent)


    for i_episode in range(num_episodes):
        # Initialize the environment and get it's state
        env.reset()
        env.render()
        state, reward, _, _, _ =env.last()
        save_state("adversary_0", state, reward)
        save_state("adversary_1", state, reward)
        save_state("adversary_2", state, reward)
        save_state("agent_0", state, reward)
        rewards = []
        actions = {i:torch.tensor([[0]], device=device) for i in ["adversary_0", "adversary_1", "adversary_2", "agent_0"]}
        for t in count():
            agent = ["adversary_0", "adversary_1", "adversary_2", "agent_0"][t%4]
            previous_state = get_state(agent)
            observation, reward, terminated, truncated, _ = env.last()
            observation, reward = deal_state(agent, observation, reward)
            rewards.append(reward)
            old_action = actions[agent]
            done = terminated or truncated
            if done:
                env.step(None)
            else:
                action = select_action(observation, good_agent=("agent" in agent))
                actions[agent] = action
                env.step(action.item())

                # Store the transition in memory
                if "agent" not in agent:
                    memory.push(previous_state[0], old_action, observation, previous_state[1])

                    # Move to the next state
                    # state = next_state

                    # Perform one step of the optimization (on the policy network)
                    optimize_model()

                    # Soft update of the target network's weights
                    # θ′ ← τ θ + (1 −τ )θ′
                    target_net_state_dict = target_net.state_dict()
                    policy_net_state_dict = policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                    target_net.load_state_dict(target_net_state_dict)
            env.render()
            if done:
                episode_durations.append(t + 1)
                episode_rewards += rewards[-4:]
                print_rewards()
                #env.close()
                break

    print('Complete')
    return torch.tensor(episode_rewards, dtype=torch.float)


batches_num = 3
task_handles = []
try:
    for i in range(1, batches_num+1):
        task_handles.append(do_the_flop.remote(name=i))

    output = ray.get(task_handles)
    print(output)
except KeyboardInterrupt:
    for i in task_handles:
        ray.cancel(i)
