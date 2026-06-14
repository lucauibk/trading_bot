"""
Seasonality features: hour of day, day of week, weekend flag.
Crypto has documented intraday and weekly patterns.
"""

import numpy as np
from datetime import datetime


FEATURE_NAMES = [
    "hour_sin",   # hour encoded as sin/cos to preserve cyclical nature
    "hour_cos",
    "dow_sin",    # day of week sin/cos
    "dow_cos",
    "is_weekend",
]


def extract(dt: datetime = None) -> np.ndarray:
    """Extract 5 seasonality features for a given datetime (UTC)."""
    if dt is None:
        dt = datetime.utcnow()

    hour = dt.hour
    dow = dt.weekday()  # 0=Monday, 6=Sunday
    is_weekend = 1.0 if dow >= 5 else 0.0

    import math
    feats = np.array([
        math.sin(2 * math.pi * hour / 24),
        math.cos(2 * math.pi * hour / 24),
        math.sin(2 * math.pi * dow / 7),
        math.cos(2 * math.pi * dow / 7),
        is_weekend,
    ], dtype=np.float32)

    return feats
