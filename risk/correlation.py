"""
BTC correlation tracking per symbol (rolling 30-day).
Used by RiskManager to prevent over-concentration when all alts are highly correlated.
"""

import logging
import time
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)

_WINDOW_CANDLES = 30 * 24  # 30 days of 1h candles


class CorrelationTracker:
    """
    Tracks rolling 30d correlation of each symbol vs BTC.
    Update daily by calling update(btc_returns, symbol_returns).
    """

    def __init__(self):
        self._correlations: Dict[str, float] = {}
        self._last_update: Dict[str, float] = {}
        self._btc_returns: Optional[pd.Series] = None

    def update_btc(self, btc_close: pd.Series):
        self._btc_returns = btc_close.pct_change().dropna().tail(_WINDOW_CANDLES)

    def update_symbol(self, symbol: str, close: pd.Series):
        if self._btc_returns is None or len(self._btc_returns) < 100:
            return
        sym_returns = close.pct_change().dropna().tail(_WINDOW_CANDLES)
        aligned = pd.concat([self._btc_returns, sym_returns], axis=1).dropna()
        if len(aligned) < 50:
            return
        corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
        self._correlations[symbol] = corr
        self._last_update[symbol] = time.time()
        logger.debug("BTC correlation %-12s: %.3f", symbol, corr)

    def get(self, symbol: str) -> float:
        return self._correlations.get(symbol, 0.5)  # default mid-level correlation

    def high_correlation_symbols(self, threshold: float = 0.85) -> List[str]:
        return [s for s, c in self._correlations.items() if c >= threshold]

    def corr_bucket_count(self, threshold: float = 0.85) -> int:
        """How many symbols are currently in the high-correlation bucket."""
        return len(self.high_correlation_symbols(threshold))
