"""
Volume Profile – Point of Control (POC), Value Area High/Low.

Used to place grid levels near high-volume price nodes instead of linspace.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_N_BUCKETS = 50
_VALUE_AREA_PCT = 0.70  # 70% of volume = value area


def compute_volume_profile(df: pd.DataFrame, n_buckets: int = _N_BUCKETS) -> dict:
    """
    df: OHLCV DataFrame
    Returns: {poc, vah, val, profile: {price: volume}}
    """
    low = df["low"].min()
    high = df["high"].max()
    if high <= low:
        mid = (high + low) / 2
        return {"poc": mid, "vah": high, "val": low, "profile": {}}

    buckets = np.linspace(low, high, n_buckets + 1)
    volume_by_bucket = np.zeros(n_buckets)

    for _, row in df.iterrows():
        price_low = row["low"]
        price_high = row["high"]
        vol = row["volume"]
        for i in range(n_buckets):
            b_low = buckets[i]
            b_high = buckets[i + 1]
            overlap = max(0, min(price_high, b_high) - max(price_low, b_low))
            price_range = price_high - price_low
            if price_range > 0 and overlap > 0:
                volume_by_bucket[i] += vol * overlap / price_range

    poc_idx = int(np.argmax(volume_by_bucket))
    poc = (buckets[poc_idx] + buckets[poc_idx + 1]) / 2

    # Value Area: expand from POC until 70% of volume covered
    total_vol = volume_by_bucket.sum()
    target = total_vol * _VALUE_AREA_PCT
    lo_idx = hi_idx = poc_idx
    area_vol = volume_by_bucket[poc_idx]

    while area_vol < target:
        expand_lo = lo_idx > 0
        expand_hi = hi_idx < n_buckets - 1
        if not expand_lo and not expand_hi:
            break
        lo_vol = volume_by_bucket[lo_idx - 1] if expand_lo else 0
        hi_vol = volume_by_bucket[hi_idx + 1] if expand_hi else 0
        if hi_vol >= lo_vol and expand_hi:
            hi_idx += 1
            area_vol += volume_by_bucket[hi_idx]
        elif expand_lo:
            lo_idx -= 1
            area_vol += volume_by_bucket[lo_idx]
        else:
            hi_idx += 1
            area_vol += volume_by_bucket[hi_idx]

    val = (buckets[lo_idx] + buckets[lo_idx + 1]) / 2
    vah = (buckets[hi_idx] + buckets[hi_idx + 1]) / 2

    profile = {
        float((buckets[i] + buckets[i + 1]) / 2): float(volume_by_bucket[i])
        for i in range(n_buckets)
    }

    logger.debug("VolumeProfile POC=%.4f VAH=%.4f VAL=%.4f", poc, vah, val)
    return {"poc": poc, "vah": vah, "val": val, "profile": profile}


def volume_weighted_levels(poc: float, vah: float, val: float,
                            n_levels: int, profile: dict) -> list:
    """
    Generate grid levels weighted toward high-volume zones.
    Levels are denser near POC (higher fill probability).
    Returns sorted list of prices.
    """
    if not profile or vah <= val:
        return list(np.linspace(val, vah, n_levels))

    prices = sorted(profile.keys())
    vols = np.array([profile[p] for p in prices])
    total = vols.sum()
    if total == 0:
        return list(np.linspace(val, vah, n_levels))

    # Cumulative distribution → invert for level placement
    cdf = np.cumsum(vols) / total
    quantiles = np.linspace(0.02, 0.98, n_levels)
    levels = []
    for q in quantiles:
        idx = int(np.searchsorted(cdf, q))
        idx = min(idx, len(prices) - 1)
        levels.append(prices[idx])

    return sorted(set(levels))
