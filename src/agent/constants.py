DEVICE = "cpu"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AGENTS = ["adversary_0", "adversary_1", "adversary_2", "agent_0"]

ACTIONS = {
    "no_action": 0,
    "move_left": 1,
    "move_right": 2,
    "move_down": 3,
    "move_up": 4,
}

import torch

device = torch.device(DEVICE)

cfg = {
    "batch_size": 128,
    "gamma": 0.99,
    "eps_start": 0.9,
    "eps_end": 0.05,
    "eps_decay": 1000,
    "tau": 0.005,
    "learning_rate": 1e-4,
    "replay_mem": 256,  # 00)  # rob: reduced to reduce RAM usage
    "ray_batches": 48,
    "eps_num": 100,
    "max_cycles": 1000,
}

test_cfg = {
    **cfg,
    "ray_batches": 3,
    "eps_num": 1,
    "max_cycles": 5000,
}

hpo_cfg = {  # testing values
    "eps_num": 6,
    "max_cycles": 1000,
    "batch_size": 256,
    "replay_mem": 256,
}


search_space_cfg = {
    "gamma": [0.9, 0.99],
    "eps_start": [0.8, 0.9],
    "eps_end": [0.02, 0.05],
    "eps_decay": [200, 2000],
    "tau": [0.005, 0.01],
    "learning_rate": [1e-4, 1e-3],
}

import optuna


def define_search_space(trial: optuna.Trial):
    trial.suggest_float("gamma", 0.9, 0.99)
    trial.suggest_float("eps_start", 0.8, 0.9)
    trial.suggest_float("eps_end", 0.02, 0.05)
    trial.suggest_float("eps_decay", 200, 2000)
    trial.suggest_float("tau", 0.005, 0.01)
    trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)


search_space_cfg_old = {
    "batch_size": [64, 128, 256],
    "gamma": [0.9, 0.95, 0.99],
    "eps_start": [0.8, 0.9],
    "eps_end": [0.02, 0.05],
    "eps_decay": [200, 500, 1000],
    "tau": [0.005, 0.008, 0.01],
    "learning_rate": [1e-3, 1e-4],
    "replay_mem": [256, 512],  # 00)  # rob: reduced to reduce RAM usage
    # "ray_batches": [48],
    # "eps_num": [100],
    # "max_cycles": [1000],
}

search_space_cfg_old2 = {
    "batch_size": [64, 256],
    "gamma": [0.99],
    "eps_start": [0.9],
    "eps_end": [0.05],
    "eps_decay": [1000],
    "tau": [0.01],
    "learning_rate": [1e-4, 1e-3],
    # "replay_mem": [256, 512],  # 00)  # rob: reduced to reduce RAM usage
    # "ray_batches": [48],
    # "eps_num": [100],
    # "max_cycles": [1000],
}
