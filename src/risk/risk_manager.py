"""
Risikomanagement: Position Sizing, Daily Drawdown, Portfolio-Limits.
"""

import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    quantity:   float
    risk_usdt:  float
    valid:      bool
    reason:     str = ""


class RiskManager:

    def __init__(self, params: dict, initial_capital: float):
        self.params          = params
        self.initial_capital = initial_capital
        self._daily_start    = initial_capital
        self._last_reset     = date.today()

    def _reset_daily_if_needed(self, capital: float):
        today = date.today()
        if today != self._last_reset:
            self._daily_start = capital
            self._last_reset  = today

    def calculate_position_size(self, capital: float, entry: float,
                                 stop_loss: float) -> PositionSize:
        max_risk   = self.params.get("max_risk_per_trade", 0.01)
        max_pos    = self.params.get("max_position_size", 0.10)
        risk_usdt  = capital * max_risk
        stop_dist  = abs(entry - stop_loss)

        if stop_dist <= 0:
            return PositionSize(0, 0, False, "Stop-Distanz = 0")

        qty = risk_usdt / stop_dist
        notional = qty * entry

        # Position nicht größer als max_position_size des Kapitals
        max_notional = capital * max_pos
        if notional > max_notional:
            qty = max_notional / entry
            notional = max_notional

        if notional < 10:
            return PositionSize(0, 0, False, f"Notional zu klein: {notional:.2f} USDT")

        return PositionSize(quantity=qty, risk_usdt=risk_usdt, valid=True)

    def check_daily_drawdown(self, capital: float) -> bool:
        """True = Trading erlaubt, False = Tageslimit erreicht."""
        self._reset_daily_if_needed(capital)
        max_dd = self.params.get("max_daily_drawdown", 0.03)
        daily_loss = (capital - self._daily_start) / self._daily_start
        if daily_loss <= -max_dd:
            logger.warning("Tages-Drawdown erreicht: %.2f%% – Bot pausiert", daily_loss * 100)
            return False
        return True

    def check_portfolio_risk(self, open_positions: list, capital: float) -> bool:
        """True = neue Position erlaubt."""
        max_portfolio = self.params.get("max_portfolio_risk", 0.05)
        max_open      = self.params.get("max_open_positions", 3)

        if len(open_positions) >= max_open:
            logger.info("Max offene Positionen erreicht (%d)", max_open)
            return False

        total_risk = sum(p.get("risk_usdt", 0) for p in open_positions)
        if total_risk / max(capital, 1) >= max_portfolio:
            logger.info("Portfolio-Risikolimit erreicht: %.2f USDT", total_risk)
            return False

        return True

    def can_trade(self, capital: float, open_positions: list) -> bool:
        return (self.check_daily_drawdown(capital) and
                self.check_portfolio_risk(open_positions, capital))
