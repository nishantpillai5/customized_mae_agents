DEVICE = "cpu"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
REPLAY_MEM = 256  # 00)  # rob: reduced to reduce RAM usage


AGENTS = ["adversary_0", "adversary_1", "adversary_2", "agent_0"]

ACTIONS = {
    "no_action": 0,
    "move_left": 1,
    "move_right": 2,
    "move_down": 3,
    "move_up": 4,
}

RAY_BATCHES = 3
EPS_NUM = 10
MAX_CYCLES = 2500

import torch

device = torch.device(DEVICE)
