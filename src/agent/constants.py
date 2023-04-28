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

# After HPO ad_tune_Apr27_1435_06134012.log
wait_cfg = {
    "batch_size": 512,
    "eps_decay": 200,
    "eps_end": 0.039167333085595155,
    "eps_start": 0.8914499472528596,
    "gamma": 0.8719405940387375,
    "learning_rate": 0.0037918476613645136,
    "num_layers": 6,
    "num_neurons": 16,
    "replay_mem": 256,
    "tau": 0.008439390442004039,
    # Non-HPO
    "strats": ["evasive", "hiding", "shifty"],
    "ray_batches": 3,
    # "eps_num": 100,
    # "max_cycles": 6000,
    "eps_num": 12,
    "max_cycles": 1000,
}

cfg = {
    "batch_size": 128,
    "gamma": 0.99,
    "eps_start": 0.9,
    "eps_end": 0.05,
    "eps_decay": 1000,
    "tau": 0.005,
    "learning_rate": 1e-4,
    "replay_mem": 256,  # 00)  # rob: reduced to reduce RAM usage
    "ray_batches": 3,
    "eps_num": 12,
    "max_cycles": 1000,
    "num_layers": 4,
    "num_neurons": 64,
    "strats": ["evasive", "hiding", "shifty"],
}

test_cfg = {
    **cfg,
    "ray_batches": 3,
    "eps_num": 1,
    "max_cycles": 5000,
}

# hpo_cfg = {  # testing values
#     "eps_num": 1,
#     "max_cycles": 100,
# }

hpo_cfg = {  # testing values
    "eps_num": 6,
    "max_cycles": 1000,
}

import optuna


def define_search_space(trial: optuna.Trial):
    trial.suggest_float("gamma", 0.8, 0.99)
    trial.suggest_float("eps_start", 0.8, 0.9)
    trial.suggest_float("eps_end", 0.02, 0.05)
    trial.suggest_int("eps_decay", 150, 2000, step=50)
    trial.suggest_float("tau", 0.005, 0.01, log=True)
    trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    trial.suggest_int("batch_size", 128, 512, step=128)
    trial.suggest_int("replay_mem", 128, 512, step=128)
    trial.suggest_int("num_layers", 3, 6, step=1)
    trial.suggest_int("num_neurons", 16, 80, step=16)


def define_search_space_test(trial: optuna.Trial):
    trial.suggest_float("gamma", 0.85, 0.95)
    trial.suggest_float("eps_start", 0.8, 0.9)
    trial.suggest_float("eps_end", 0.03, 0.045)
    trial.suggest_int("eps_decay", 180, 220, step=10)
    trial.suggest_float("tau", 0.007, 0.009, log=True)
    trial.suggest_float("learning_rate", 7e-4, 5e-3, log=True)
    trial.suggest_int("batch_size", 256, 512, step=256)
    trial.suggest_int("replay_mem", 256, 512, step=256)
    trial.suggest_int("num_layers", 5, 8, step=1)
    trial.suggest_int("num_neurons", 16, 64, step=16)


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
