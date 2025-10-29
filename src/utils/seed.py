import os
import random
import numpy as np


def set_global_seed(seed: int) -> None:
    """Set seeds for Python, NumPy, and hash to ensure reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)



