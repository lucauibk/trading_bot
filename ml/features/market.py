"""
Market context features: BTC returns, BTC-alt correlation, BTC dominance.
Alts are highly correlated with BTC – ignoring this leads to over-fitting on alt signals.
"""

import numpy as np
from typing import Optional

from core.context import BTCContext

FEATURE_NAMES = [
    "btc_return_1h",
    "btc_return_4h",
    "btc_return_24h",
    "btc_corr_30d",
    "btc_dominance",
]


def extract(btc: Optional[BTCContext], btc_corr: float = 0.0) -> np.ndarray:
    """Extract 5 market context features.

    btc_corr_30d is computed independently from OHLCV and preserved even without a
    BTCContext. The neutral fallback is 0.0 — matching both production callers
    (ml/predictor.py and ml/trainer.py default to 0.0), not a hardcoded 0.5 guess.
    """
    if btc is None:
        feats = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        feats[3] = np.clip(btc_corr, -1.0, 1.0)
        return feats

    feats = np.array([
        np.clip(btc.return_1h, -0.10, 0.10),
        np.clip(btc.return_4h, -0.20, 0.20),
        np.clip(btc.return_24h, -0.40, 0.40),
        np.clip(btc_corr, -1.0, 1.0),
        np.clip(btc.dominance, 0.0, 1.0),
    ], dtype=np.float32)

    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
