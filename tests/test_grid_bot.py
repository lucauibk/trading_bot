"""
Unit-Tests für Grid Bot: _calc_level_allocations, _maybe_compound,
check_stop_loss, Per-Position-Stop-Loss, RiskManager-Integration.

Aufruf: python3 -m pytest tests/test_grid_bot.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pytest
from unittest.mock import MagicMock, patch


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_paper_bot(investment=300.0, levels=6):
    """Erstellt PaperGridBot ohne externe Abhängigkeiten."""
    import config as cfg
    # Minimal-Mocks damit Imports nicht fehlschlagen
    with patch("notifier.notify_trade_close"), \
         patch("notifier._send"):
        from grid_bot import PaperGridBot
        bot = PaperGridBot.__new__(PaperGridBot)
        bot.symbol = "SOL/USD"
        bot.investment = investment
        bot._initial_investment = investment
        bot.levels = levels
        bot.range_pct = 0.05
        bot.grid_lines = []
        bot.orders = {}
        bot.total_profit = 0.0
        bot.trade_count = 0
        bot.usdt_per_grid = investment / levels
        bot._last_compound_at = 0
        bot._compounded_profit = 0.0
        bot._direction_score = 0.0
        bot._level_allocations = {}
        bot.with_position = True
        bot._last_regime = ""
        bot._last_confidence = 0.0
        bot._last_pred_low = 0.0
        bot._last_pred_high = 0.0
        return bot


# ── _calc_level_allocations ───────────────────────────────────────────────────

class TestCalcLevelAllocations:

    def test_neutral_score_equal_distribution(self):
        from grid_bot import _calc_level_allocations
        levels = [90.0, 95.0, 100.0, 105.0, 110.0]
        allocs = _calc_level_allocations(levels, 100.0, 500.0, direction_score=0.0)
        assert len(allocs) == 5
        for p in levels:
            assert abs(allocs[p] - 100.0) < 1e-6, "Neutral score → gleichmäßige Verteilung"

    def test_bullish_score_more_budget_at_lower_levels(self):
        from grid_bot import _calc_level_allocations
        levels = [85.0, 90.0, 95.0, 105.0, 110.0, 115.0]
        allocs = _calc_level_allocations(levels, 100.0, 600.0, direction_score=1.0)
        assert allocs[85.0] > allocs[115.0], "Bullish: tiefe Level > hohe Level"
        assert abs(sum(allocs.values()) - 600.0) < 1e-4

    def test_bearish_score_more_budget_at_upper_levels(self):
        from grid_bot import _calc_level_allocations
        levels = [85.0, 90.0, 95.0, 105.0, 110.0, 115.0]
        allocs = _calc_level_allocations(levels, 100.0, 600.0, direction_score=-1.0)
        assert allocs[115.0] > allocs[85.0], "Bearish: hohe Level > tiefe Level"
        assert abs(sum(allocs.values()) - 600.0) < 1e-4

    def test_single_level_no_crash(self):
        from grid_bot import _calc_level_allocations
        allocs = _calc_level_allocations([100.0], 100.0, 300.0, direction_score=0.8)
        assert abs(allocs[100.0] - 300.0) < 1e-4


# ── _maybe_compound ────────────────────────────────────────────────────────────

class TestMaybeCompound:

    def test_no_compound_when_no_profit(self):
        bot = _make_paper_bot(investment=300.0)
        bot.total_profit = -5.0
        bot.trade_count = 10
        original = bot.investment
        with patch("grid_bot.notifier._send"):
            bot._maybe_compound(current_price=100.0)
        assert bot.investment == original, "Kein Compound bei Verlust"

    def test_no_compound_before_threshold(self):
        bot = _make_paper_bot(investment=300.0)
        bot.total_profit = 10.0
        bot.trade_count = 3  # < COMPOUND_EVERY_TRADES=5
        original = bot.investment
        with patch("grid_bot.notifier._send"):
            bot._maybe_compound(current_price=100.0)
        assert bot.investment == original, "Kein Compound vor Schwelle"

    def test_compound_adds_profit_delta(self):
        bot = _make_paper_bot(investment=300.0)
        bot.total_profit = 20.0
        bot._compounded_profit = 0.0
        bot.trade_count = 5
        bot._last_compound_at = 0
        with patch("grid_bot.notifier._send"), \
             patch.object(bot, "setup_grid"):
            bot._maybe_compound(current_price=100.0)
        assert abs(bot.investment - 320.0) < 1e-4, "Investment soll um Profit-Delta steigen"

    def test_compound_respects_investment_cap(self):
        from grid_bot import MAX_INVESTMENT_MULT
        bot = _make_paper_bot(investment=300.0)
        cap = 300.0 * MAX_INVESTMENT_MULT
        bot.total_profit = 5000.0   # weit über Cap
        bot._compounded_profit = 0.0
        bot.trade_count = 5
        bot._last_compound_at = 0
        with patch("grid_bot.notifier._send"), \
             patch.object(bot, "setup_grid"):
            bot._maybe_compound(current_price=100.0)
        assert bot.investment <= cap + 1e-4, f"Investment darf {cap} nicht überschreiten"

    def test_compound_only_delta_not_cumulative(self):
        bot = _make_paper_bot(investment=300.0)
        bot.total_profit = 30.0
        bot._compounded_profit = 20.0  # 20 bereits reinvestiert
        bot.trade_count = 10
        bot._last_compound_at = 5
        with patch("grid_bot.notifier._send"), \
             patch.object(bot, "setup_grid"):
            bot._maybe_compound(current_price=100.0)
        assert abs(bot.investment - 310.0) < 1e-4, "Nur Delta (10) soll reinvestiert werden"


# ── check_stop_loss ────────────────────────────────────────────────────────────

class TestCheckStopLoss:

    def test_no_stop_loss_within_limit(self):
        bot = _make_paper_bot(investment=300.0)
        bot.total_profit = -10.0   # 3.3% < 8%
        with patch("grid_bot.notifier._send"):
            assert not bot.check_stop_loss()

    def test_stop_loss_triggered_at_8pct(self):
        bot = _make_paper_bot(investment=300.0)
        bot.total_profit = -24.1   # > 8% von 300
        with patch("grid_bot.notifier._send"):
            assert bot.check_stop_loss()

    def test_stop_loss_scales_with_investment(self):
        bot = _make_paper_bot(investment=600.0)  # nach Compounding
        bot.total_profit = -40.0   # 6.7% < 8% → kein Stop
        with patch("grid_bot.notifier._send"):
            assert not bot.check_stop_loss()

        bot.total_profit = -50.0   # 8.3% > 8% → Stop
        with patch("grid_bot.notifier._send"):
            assert bot.check_stop_loss()

    def test_stop_loss_at_exact_boundary(self):
        bot = _make_paper_bot(investment=300.0)
        # max_loss = 300 * 0.08 = 24.0; Bedingung: total_profit <= -max_loss → -24.0 triggert
        bot.total_profit = -23.99  # knapp unter Grenze → kein Stop
        with patch("grid_bot.notifier._send"):
            assert not bot.check_stop_loss()

        bot.total_profit = -24.0  # exakt Grenze → Stop triggert
        with patch("grid_bot.notifier._send"):
            assert bot.check_stop_loss()


# ── Per-Position Stop-Loss ─────────────────────────────────────────────────────

class TestPerPositionStopLoss:

    def _bot_with_open_sell(self, buy_price: float, qty: float = 1.0):
        """Erstellt Bot mit einer offenen Sell-Order inkl. SL."""
        from grid_bot import PER_POS_SL_PCT
        bot = _make_paper_bot()
        sl_price = buy_price * (1 - PER_POS_SL_PCT)
        sell_price = buy_price * 1.02
        bot.grid_lines = [buy_price, sell_price]
        bot.orders = {
            sell_price: {
                "side": "sell",
                "qty": qty,
                "filled": False,
                "bought_at": buy_price,
                "sl_price": sl_price,
            }
        }
        return bot, sell_price, sl_price

    def test_sl_not_triggered_above_sl_price(self):
        bot, sell_price, sl_price = self._bot_with_open_sell(buy_price=100.0)
        current_price = sl_price + 1.0  # über SL
        with patch("grid_bot.notifier._send"), \
             patch("grid_bot.config") as mock_cfg:
            mock_cfg.PAPER_TRADING = True
            with patch("dashboard.db.log_trade"):
                bot._check_position_stop_losses(current_price)
        assert not bot.orders[sell_price]["filled"], "SL darf nicht bei Preis über SL-Level triggern"

    def test_sl_triggered_at_sl_price(self):
        from grid_bot import PER_POS_SL_PCT, KRAKEN_FEE
        buy_price = 100.0
        bot, sell_price, sl_price = self._bot_with_open_sell(buy_price=buy_price, qty=2.0)
        with patch("grid_bot.notifier._send"), \
             patch("grid_bot.config") as mock_cfg:
            mock_cfg.PAPER_TRADING = True
            with patch("dashboard.db.log_trade"):
                bot._check_position_stop_losses(sl_price)
        assert bot.orders[sell_price]["filled"], "SL soll bei current_price ≤ sl_price triggern"
        expected_loss = (sl_price - buy_price) * 2.0 - (sl_price + buy_price) * 2.0 * KRAKEN_FEE
        assert abs(bot.total_profit - expected_loss) < 1e-4

    def test_sl_correct_pnl_negative(self):
        from grid_bot import PER_POS_SL_PCT
        buy_price = 200.0
        bot, sell_price, sl_price = self._bot_with_open_sell(buy_price=buy_price)
        with patch("grid_bot.notifier._send"), \
             patch("grid_bot.config") as mock_cfg:
            mock_cfg.PAPER_TRADING = True
            with patch("dashboard.db.log_trade"):
                bot._check_position_stop_losses(sl_price)
        assert bot.total_profit < 0, "Stop-Loss soll negativen PnL produzieren"

    def test_check_fills_calls_sl_check(self):
        """check_fills ruft _check_position_stop_losses auf."""
        bot = _make_paper_bot()
        bot.grid_lines = []
        bot.orders = {}
        with patch.object(bot, "_check_position_stop_losses") as mock_sl:
            bot.check_fills(current_price=100.0)
        mock_sl.assert_called_once_with(100.0)
