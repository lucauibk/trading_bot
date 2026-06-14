"""
Perp/Futures features: funding rate, funding z-score, OI changes.
These are among the strongest directional signals in crypto.

Positive funding z-score: longs crowded → potential reversal or continued momentum.
High OI + rising price: strong trend confirmation.
"""

import numpy as np
from typing import Optional

from core.context import FundingInfo

FEATURE_NAMES = [
    "funding_rate",
    "funding_z7d",
    "oi_change_1h",
    "oi_change_24h",
]


def extract(funding: Optional[FundingInfo]) -> np.ndarray:
    """Extract 4 perp features from FundingInfo. Returns zeros if unavailable."""
    if funding is None:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    feats = np.array([
        np.clip(funding.rate * 1000, -5.0, 5.0),      # scale to ~[-5, 5]
        np.clip(funding.rate_z7d, -3.0, 3.0),          # z-score
        np.clip(funding.oi_change_1h, -0.20, 0.20),    # ±20%
        np.clip(funding.oi_change_24h, -0.50, 0.50),   # ±50%
    ], dtype=np.float32)

    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
