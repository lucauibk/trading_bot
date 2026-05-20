from typing import List
import numpy as np


def compute_grid_levels(low: float, high: float, count: int = 10) -> List[float]:
    if count < 2:
        raise ValueError("count must be >= 2")
    if low >= high:
        raise ValueError("low must be < high")
    levels = np.linspace(low, high, count)
    return [round(float(v), 8) for v in levels]
