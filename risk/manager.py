"""
RiskManager – reads config/config.yaml and enforces all risk rules.

Pre-trade checks via can_open() replace the scattered inline checks in grid_bot.py.
Daily drawdown is cross-coin, loaded from config.yaml (aktuell 10%, bewusste
Entscheidung dd0531b – config.yaml ist führend). Correlation-aware bucketing
prevents 5-alt over-concentration.
"""

import logging
from datetime import date
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
        # config.yaml ist führend (aktuell 0.10, bewusste Entscheidung dd0531b);
        # Default hier nur Fallback falls die yaml fehlt – gleicher Wert, damit
        # Code-Default und Config nicht divergieren (#40).
        self.max_daily_drawdown: float = cfg.get("max_daily_drawdown", 0.10)
        self.max_position_size: float = cfg.get("max_position_size", 0.10)
        self.max_open_positions: int = cfg.get("max_open_positions", 5)
        self.max_portfolio_risk: float = cfg.get("max_portfolio_risk", 0.05)
        self.max_corr_bucket: int = 2  # max coins from high-corr bucket open simultaneously

        self.corr = correlation

        self._daily_start: float = 0.0
        self._daily_date: date = date.today()

        logger.info(
            "RiskManager loaded: max_dd=%.1f%% max_pos=%.0f%% max_open=%d",
            self.max_daily_drawdown * 100,
            self.max_position_size * 100,
            self.max_open_positions,
        )

    def set_daily_start(self, equity: float):
        today = date.today()
        if today != self._daily_date:
            self._daily_start = equity
            self._daily_date = today
        elif self._daily_start == 0.0:
            self._daily_start = equity

    def daily_drawdown_ok(self, equity: float) -> bool:
        if self._daily_start <= 0:
            return True
        dd = (equity - self._daily_start) / self._daily_start
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
            # Fail-closed: Equity 0/negativ heißt "noch nicht initialisiert"
            # (oder Totalverlust) – vorher wurden hier ALLE Checks übersprungen
            # und Entries liefen ungeprüft durch (#36).
            return False, "equity_uninitialized"

        # 1. Daily drawdown
        if not self.daily_drawdown_ok(equity):
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

        Bewusst OHNE leverage-Parameter: compute_position_usdt kennt keinen
        Hebel, ein früherer leverage-Arg wurde still verworfen (#53).
        """
        return compute_position_usdt(
            equity=equity,
            win_rate=win_rate,
            win_loss_ratio=win_loss_ratio,
            realized_vol=realized_vol,
            target_risk_pct=0.01,
            max_position_pct=self.max_position_size,
        )
