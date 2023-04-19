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
    "ray_batches": 2,
    "eps_num": 150,
    "max_cycles": 400,
}

test_cfg = {
    **cfg,
    "ray_batches": 3,
    "eps_num": 1,
    "max_cycles": 5000,
}
