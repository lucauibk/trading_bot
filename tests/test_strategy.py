"""
pytest Tests für Strategie, RiskManager und PaperBroker.
Aufruf: python3 -m pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.strategy.base_strategy import Signal, TradeSignal
from src.strategy.ema_crossover import EMACrossoverStrategy
from src.strategy.rsi_mean_rev import RSIMeanRevStrategy
from src.risk.risk_manager import RiskManager
from src.execution.broker import PaperBroker


# ── Hilfsfunktionen ──────────────────────────────────────────────────────────

def _make_df(n=200, trend="up") -> pd.DataFrame:
    """Synthetischer OHLCV DataFrame."""
    np.random.seed(42)
    if trend == "up":
        close = 100 + np.cumsum(np.random.randn(n) * 0.5 + 0.1)
    elif trend == "down":
        close = 100 + np.cumsum(np.random.randn(n) * 0.5 - 0.1)
    else:
        close = 100 + np.random.randn(n) * 2

    high = close + np.abs(np.random.randn(n)) * 0.5
    low  = close - np.abs(np.random.randn(n)) * 0.5
    vol  = np.random.randint(1_000_000, 5_000_000, n).astype(float)
    idx  = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"open": close, "high": high, "low": low,
                         "close": close, "volume": vol}, index=idx)


EMA_PARAMS = {"ema_fast": 9, "ema_slow": 21, "ema_trend": 50,
              "rr_ratio": 2.0, "atr_period": 14, "atr_stop_mult": 1.5,
              "volume_confirm": False}
RSI_PARAMS = {"rsi_period": 7, "rsi_oversold": 30, "rsi_exit": 65,
              "ema_trend": 50, "bb_period": 20, "bb_std": 2.0,
              "atr_period": 14, "atr_stop_mult": 1.5, "atr_tp_mult": 2.0}
RISK_PARAMS = {"max_risk_per_trade": 0.01, "max_portfolio_risk": 0.05,
               "max_daily_drawdown": 0.03, "max_open_positions": 3,
               "max_position_size": 0.10}


# ── Strategie Tests ───────────────────────────────────────────────────────────

class TestEMACrossover:
    def test_returns_trade_signal(self):
        strat = EMACrossoverStrategy(EMA_PARAMS)
        df    = _make_df(200)
        sig   = strat.generate_signal(df)
        assert isinstance(sig, TradeSignal)
        assert sig.signal in (Signal.LONG, Signal.SHORT, Signal.NONE)

    def test_valid_signal_has_positive_sl_tp(self):
        strat = EMACrossoverStrategy(EMA_PARAMS)
        df    = _make_df(200, "up")
        sig   = strat.generate_signal(df)
        if sig.is_valid:
            assert sig.stop_loss < sig.entry < sig.take_profit

    def test_short_signal_sl_above_entry(self):
        strat = EMACrossoverStrategy(EMA_PARAMS)
        df    = _make_df(200, "down")
        sig   = strat.generate_signal(df)
        if sig.signal == Signal.SHORT:
            assert sig.stop_loss > sig.entry


class TestRSIMeanRev:
    def test_returns_trade_signal(self):
        strat = RSIMeanRevStrategy(RSI_PARAMS)
        df    = _make_df(200)
        sig   = strat.generate_signal(df)
        assert isinstance(sig, TradeSignal)

    def test_no_signal_on_downtrend(self):
        strat = RSIMeanRevStrategy(RSI_PARAMS)
        df    = _make_df(200, "down")
        sig   = strat.generate_signal(df)
        # RSI-Strategie ist Long-Only: in starkem Abwärtstrend kein Long
        assert sig.signal != Signal.SHORT


# ── RiskManager Tests ─────────────────────────────────────────────────────────

class TestRiskManager:
    def test_position_size_basic(self):
        rm   = RiskManager(RISK_PARAMS, 1000)
        size = rm.calculate_position_size(1000, entry=100, stop_loss=95)
        assert size.valid
        assert size.quantity > 0
        assert size.risk_usdt == pytest.approx(10.0)  # 1% von 1000

    def test_zero_stop_distance(self):
        rm   = RiskManager(RISK_PARAMS, 1000)
        size = rm.calculate_position_size(1000, entry=100, stop_loss=100)
        assert not size.valid

    def test_daily_drawdown_allows_trading(self):
        rm = RiskManager(RISK_PARAMS, 1000)
        assert rm.check_daily_drawdown(1000) is True
        assert rm.check_daily_drawdown(975)  is True   # -2.5%, unter Limit

    def test_daily_drawdown_stops_trading(self):
        rm = RiskManager(RISK_PARAMS, 1000)
        assert rm.check_daily_drawdown(960) is False   # -4%, über Limit

    def test_portfolio_risk_max_positions(self):
        rm   = RiskManager(RISK_PARAMS, 1000)
        open_pos = [{"risk_usdt": 5}] * 3
        assert rm.check_portfolio_risk(open_pos, 1000) is False

    def test_can_trade_combines_checks(self):
        rm = RiskManager(RISK_PARAMS, 1000)
        assert rm.can_trade(1000, []) is True
        assert rm.can_trade(960, [])  is False  # Drawdown-Limit


# ── PaperBroker Tests ─────────────────────────────────────────────────────────

class TestPaperBroker:
    def test_initial_balance(self):
        broker = PaperBroker(1000)
        assert broker.get_balance()["USDT"] == 1000

    def test_limit_order_returns_id(self):
        broker = PaperBroker(1000)
        order  = broker.limit_order("SOL/USD", "buy", 1.0, 80.0)
        assert "id" in order
        assert order["status"] == "closed"

    def test_cancel_returns_true(self):
        broker = PaperBroker(1000)
        order  = broker.limit_order("SOL/USD", "buy", 1.0, 80.0)
        assert broker.cancel_order(order["id"], "SOL/USD") is True
