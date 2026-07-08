"""
Combined feature vector: technical (16) + perp (4) + market (5) + htf (4) + seasonality (5) = 34 features.

New models should use extract_all(); the legacy extract_features() still works for backward compat.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

from core.context import BTCContext, FundingInfo
from ml.features import technical, perp, market, htf, seasonality

ALL_FEATURE_NAMES = (
    technical.FEATURE_NAMES
    + perp.FEATURE_NAMES
    + market.FEATURE_NAMES
    + htf.FEATURE_NAMES
    + seasonality.FEATURE_NAMES
)

N_FEATURES = len(ALL_FEATURE_NAMES)


def extract_all(
    df_1h: pd.DataFrame,
    funding: Optional[FundingInfo] = None,
    btc: Optional[BTCContext] = None,
    btc_corr: float = 0.0,
    dt: Optional[datetime] = None,
) -> np.ndarray:
    """
    Extract the full 34-feature vector.
    Falls back gracefully if perp/market/btc data unavailable.
    """
    tech = technical.extract(df_1h)
    perp_feats = perp.extract(funding)
    market_feats = market.extract(btc, btc_corr)
    htf_feats = htf.extract(df_1h)
    season_feats = seasonality.extract(dt)

    combined = np.concatenate([tech, perp_feats, market_feats, htf_feats, season_feats])
    return np.nan_to_num(combined.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
