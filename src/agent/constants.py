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

EPS_NUM = 10

AGENTS = ["adversary_0", "adversary_1", "adversary_2", "agent_0"]

RAY_BATCHES = 3

MAX_CYCLES = 400

import torch

device = torch.device(DEVICE)
