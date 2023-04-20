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


search_space_cfg = {
    "batch_size": [128],
    "gamma": [0.99],
    "eps_start": [0.9],
    "eps_end": [0.05],
    "eps_decay": [1000],
    "tau": [0.005],
    "learning_rate": [1e-4],
    "replay_mem": [256],  # 00)  # rob: reduced to reduce RAM usage
    "ray_batches": [48],
    "eps_num": [100],
    "max_cycles": [1000],
}

# config = {
#     "uniform": tune.uniform(-5, -1),  # Uniform float between -5 and -1
#     "quniform": tune.quniform(3.2, 5.4, 0.2),  # Round to multiples of 0.2
#     "loguniform": tune.loguniform(1e-4, 1e-1),  # Uniform float in log space
#     "qloguniform": tune.qloguniform(1e-4, 1e-1, 5e-5),  # Round to multiples of 0.00005
#     "randn": tune.randn(10, 2),  # Normal distribution with mean 10 and sd 2
#     "qrandn": tune.qrandn(10, 2, 0.2),  # Round to multiples of 0.2
#     "randint": tune.randint(-9, 15),  # Random integer between -9 and 15
#     "qrandint": tune.qrandint(-21, 12, 3),  # Round to multiples of 3 (includes 12)
#     "lograndint": tune.lograndint(1, 10),  # Random integer in log space
#     "qlograndint": tune.qlograndint(1, 10, 2),  # Round to multiples of 2
#     "choice": tune.choice(["a", "b", "c"]),  # Choose one of these options uniformly
#     "func": tune.sample_from(
#         lambda spec: spec.config.uniform * 0.01
#     ),  # Depends on other value
#     "grid": tune.grid_search([32, 64, 128]),  # Search over all these values
# }
