import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Make experiments reproducible.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")
