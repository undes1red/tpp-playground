import torch, random
import numpy as np


# Referring to https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)