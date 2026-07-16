"""
RiskManager – reads config/config.yaml and enforces all risk rules.

Pre-trade checks via can_open() replace the scattered inline checks in grid_bot.py.
The cross-coin drawdown brake is anchored to the deposit baseline (the configured
starting capital), loaded from config.yaml (`max_daily_drawdown`, was hardcoded 8%).
It is NOT a daily-resetting brake: the engine feeds a constant deposit baseline, so
there is no intraday/daily reset or relief — the brake measures drawdown from the
initial deposit (see #132).
Correlation-aware bucketing prevents 5-alt over-concentration.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import yaml

from core.context import MarketContext
from risk.correlation import CorrelationTracker
from risk.sizing import compute_position_usdt

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path("config/config.yaml")


def _load_risk_config() -> dict:
    try:
        with open(_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("risk", {})
    except Exception as e:
        logger.warning("Could not load config.yaml risk section: %s – using defaults", e)
        return {}


class RiskManager:

    def __init__(self, correlation: CorrelationTracker):
        cfg = _load_risk_config()
        self.max_daily_drawdown: float = cfg.get("max_daily_drawdown", 0.03)
        self.max_position_size: float = cfg.get("max_position_size", 0.10)
        self.max_open_positions: int = cfg.get("max_open_positions", 5)
        self.max_portfolio_risk: float = cfg.get("max_portfolio_risk", 0.05)
        self.max_corr_bucket: int = 2  # max coins from high-corr bucket open simultaneously

        self.corr = correlation

        # Drawdown baseline: anchored once to the deposit (configured starting
        # capital) by the engine. Deliberately constant — no daily reset (#132).
        self._baseline: float = 0.0

        logger.info(
            "RiskManager loaded: max_dd=%.1f%% max_pos=%.0f%% max_open=%d",
            self.max_daily_drawdown * 100,
            self.max_position_size * 100,
            self.max_open_positions,
        )

    def set_drawdown_baseline(self, equity: float):
        """Anchor the drawdown baseline once, to the deposit (starting capital).

        Set-once semantics: the first non-zero call fixes the baseline and every
        later call is a no-op. The engine feeds a constant deposit baseline every
        tick, so there is intentionally no daily/intraday reset or relief — the
        brake always measures drawdown from the original deposit (#132). A later
        equity value never re-anchors the baseline.
        """
        if self._baseline == 0.0 and equity > 0:
            self._baseline = equity

    def drawdown_ok(self, equity: float) -> bool:
        """True while equity is within max_daily_drawdown of the deposit baseline."""
        if self._baseline <= 0:
            return True
        dd = (equity - self._baseline) / self._baseline
        return dd > -self.max_daily_drawdown

    def can_open(
        self,
        symbol: str,
        usdt_size: float,
        ctx: MarketContext,
    ) -> Tuple[bool, str]:
        """
        Returns (allowed, reason).
        Checks: daily DD, max open positions, position size cap, correlation bucket.
        """
        equity = ctx.total_equity
        if equity <= 0:
            return True, ""

        # 1. Drawdown-from-deposit brake
        if not self.drawdown_ok(equity):
            return False, "daily_drawdown_exceeded"

        # 2. BTC hard filter
        btc = ctx.get_btc()
        if btc is not None:
            if btc.return_1h < -0.03:
                return False, f"btc_crash_1h({btc.return_1h:.1%})"
            if btc.realized_vol_7d > 1.5:  # annualized >150% = extreme
                return False, "btc_vol_regime_crash"

        # 3. Max open positions
        if ctx.open_position_count() >= self.max_open_positions:
            return False, f"max_open_positions({self.max_open_positions})"

        # 4. Position size cap
        if usdt_size > equity * self.max_position_size:
            return False, f"position_too_large(>{self.max_position_size:.0%})"

        # 5. Correlation bucket
        high_corr = self.corr.high_correlation_symbols(threshold=0.85)
        if symbol in high_corr:
            open_in_bucket = sum(
                1 for s in high_corr
                if ctx.symbol_position_usdt(s) > 0
            )
            if open_in_bucket >= self.max_corr_bucket:
                return False, f"corr_bucket_full({open_in_bucket}/{self.max_corr_bucket})"

        return True, ""

    def position_size(
        self,
        symbol: str,
        equity: float,
        win_rate: float = 0.52,
        win_loss_ratio: float = 1.2,
        realized_vol: float = 0.03,
    ) -> float:
        """Return recommended USDT position size using Fractional Kelly + Vol-targeting.

        Note: no `leverage` parameter. `compute_position_usdt` does not model
        leverage, so accepting one here would silently discard it and return the
        same size for any value — a trap for future callers (removed)."""
        return compute_position_usdt(
            equity=equity,
            win_rate=win_rate,
            win_loss_ratio=win_loss_ratio,
            realized_vol=realized_vol,
            target_risk_pct=0.01,
            max_position_pct=self.max_position_size,
        )
