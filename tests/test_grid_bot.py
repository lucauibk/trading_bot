"""
Tests for the new modular trading bot architecture.
Run: python3 -m pytest tests/ -v
"""

import time
import uuid

import numpy as np
import pandas as pd
import pytest


# ── PaperBroker ───────────────────────────────────────────────────────────────

class TestPaperBroker:

    def _broker(self, balance=1000.0):
        from execution.paper import PaperBroker
        return PaperBroker(initial_balance=balance)

    def test_buy_fills_when_price_hits(self):
        broker = self._broker()
        broker.place_limit("SOL/USD", "buy", 100.0, 1.0, client_id="b1")
        fills = broker.update_price("SOL/USD", 100.0)
        assert len(fills) == 1
        assert fills[0].side == "buy"
        assert fills[0].price > 100.0  # slippage applied

    def test_sell_fills_when_price_hits(self):
        broker = self._broker()
        broker.place_limit("SOL/USD", "sell", 110.0, 1.0, client_id="s1")
        broker.update_price("SOL/USD", 100.0)  # advance tick
        fills = broker.update_price("SOL/USD", 110.0)
        assert len(fills) == 1
        assert fills[0].side == "sell"

    def test_slippage_applied(self):
        broker = self._broker()
        broker.place_limit("SOL/USD", "buy", 100.0, 1.0, client_id="b1")
        fills = broker.update_price("SOL/USD", 100.0)
        assert len(fills) == 1
        assert fills[0].price > 100.0  # buy: price + slippage

    def test_balance_reduced_on_buy(self):
        broker = self._broker(balance=1000.0)
        broker.place_limit("SOL/USD", "buy", 100.0, 1.0, client_id="b1")
        broker.update_price("SOL/USD", 100.0)
        assert broker.get_balance() < 1000.0

    def test_cancel_removes_order(self):
        broker = self._broker()
        broker.place_limit("SOL/USD", "buy", 100.0, 1.0, client_id="b1")
        result = broker.cancel("b1")
        assert result is True
        fills = broker.update_price("SOL/USD", 99.0)
        assert len(fills) == 0

    def test_insufficient_balance_no_fill(self):
        broker = self._broker(balance=50.0)
        broker.place_limit("SOL/USD", "buy", 100.0, 1.0, client_id="b1")
        fills = broker.update_price("SOL/USD", 99.0)
        assert len(fills) == 0  # can't afford 100 USDT order with 50 balance


# ── Risk Sizing ───────────────────────────────────────────────────────────────

class TestRiskSizing:

    def test_kelly_fraction_basic(self):
        from risk.sizing import kelly_fraction
        f = kelly_fraction(0.55, 1.3, kelly_factor=0.25)
        assert 0 < f <= 0.25

    def test_kelly_fraction_bad_winrate(self):
        from risk.sizing import kelly_fraction
        f = kelly_fraction(0.0, 1.0)
        assert f > 0  # returns floor value

    def test_position_size_scales_with_equity(self):
        from risk.sizing import compute_position_usdt
        size_1k = compute_position_usdt(1000, 0.55, 1.3, 0.03)
        size_2k = compute_position_usdt(2000, 0.55, 1.3, 0.03)
        assert size_2k > size_1k

    def test_position_size_smaller_with_higher_vol(self):
        from risk.sizing import vol_target_size
        size_low_vol = vol_target_size(1000, 0.01, 0.02, 100.0)
        size_high_vol = vol_target_size(1000, 0.01, 0.08, 100.0)
        assert size_low_vol > size_high_vol

    def test_position_size_capped(self):
        from risk.sizing import compute_position_usdt
        size = compute_position_usdt(1000, 0.99, 10.0, 0.001, max_position_pct=0.10)
        assert size <= 100.0  # max 10% of 1000


# ── ML Features ──────────────────────────────────────────────────────────────

def _make_df(n=100):
    np.random.seed(42)
    prices = 100 * np.cumprod(1 + np.random.normal(0, 0.01, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({
        "open":   prices * 0.999,
        "high":   prices * 1.003,
        "low":    prices * 0.997,
        "close":  prices,
        "volume": np.random.uniform(1000, 5000, n),
    }, index=idx)


class TestMLFeatures:

    def test_technical_features_shape(self):
        from ml.features.technical import extract, FEATURE_NAMES
        df = _make_df()
        feats = extract(df)
        assert feats.shape == (len(FEATURE_NAMES),)
        assert not np.isnan(feats).any()

    def test_combined_features_shape(self):
        from ml.features.combined import extract_all, N_FEATURES
        df = _make_df()
        feats = extract_all(df)
        assert feats.shape == (N_FEATURES,)
        assert feats.shape == (34,)
        assert not np.isnan(feats).any()

    def test_perp_features_zeros_when_no_data(self):
        from ml.features.perp import extract
        feats = extract(None)
        assert (feats == 0).all()

    def test_market_features_zeros_when_no_data(self):
        from ml.features.market import extract
        # btc_corr_30d (index 3) is computed independently from OHLCV and is
        # preserved even without BTC context — pass a neutral 0.0 to get all-zeros.
        feats = extract(None, btc_corr=0.0)
        assert (feats == 0).all()

    def test_market_features_preserve_btc_corr_when_no_btc(self):
        from ml.features.market import extract
        # Without BTC context, only btc_corr_30d survives (default 0.5).
        feats = extract(None)
        assert feats[3] == 0.5
        assert (np.delete(feats, 3) == 0).all()

    def test_seasonality_features_cyclic(self):
        from ml.features.seasonality import extract
        from datetime import datetime
        feats_midnight = extract(datetime(2024, 1, 1, 0, 0))
        feats_noon = extract(datetime(2024, 1, 1, 12, 0))
        assert feats_midnight[0] != feats_noon[0]

    def test_htf_features_no_crash(self):
        from ml.features.htf import extract
        df = _make_df(200)
        feats = extract(df)
        assert feats.shape[0] == 4
        assert not np.isnan(feats).any()

    def test_predict_maps_proba_index_to_real_class_label(self):
        # Regression for #88: when a class is absent from training, clf.classes_
        # is not [0,1,2]. predict() must translate the positional argmax back to
        # the real class label, not return the column index.
        from ml.model import TradingModel

        class _StubClf:
            classes_ = np.array([0, 2])  # class 1 (hold) never seen in training

            def predict_proba(self, x):
                # highest proba is column 1 -> real class 2 (buy), not label 1 (hold)
                return np.array([[0.2, 0.8]])

        m = TradingModel("TEST/USD")
        m._clf = _StubClf()
        m._n_samples = m.MIN_SAMPLES
        m._feature_names = []  # skip feature-count check
        label, conf = m.predict(np.zeros(34, dtype=np.float32))
        assert label == 2  # buy, not the positional index 1 (hold)
        assert conf == 0.8


class TestReconcilerOrderCleanup:
    """Regression for #134: the live reconciler's open_orders table must be pruned
    on cancel and on fill — remove_order() had zero callers, leaking the table."""

    class _StubReconciler:
        def __init__(self):
            self.removed = []
            self._fills = []

        def reconcile(self):
            return self._fills

        def remove_order(self, client_id):
            self.removed.append(client_id)

        def track_order(self, *a, **k):
            pass

        def get_tracked_orders(self, symbol=None):
            return []

    class _StubBroker:  # deliberately NOT a PaperBroker so live paths run
        def __init__(self):
            self.cancelled = []

        def cancel(self, cid):
            self.cancelled.append(cid)

    class _StubStrategy:
        _broker = None

        def desired_orders(self, symbol, price, ctx):
            return []  # nothing desired → existing active order gets cancelled

        def on_fill(self, fill, ctx):
            pass

    def _engine(self, reconciler):
        from core.engine import Engine
        return Engine(self._StubStrategy(), self._StubBroker(), ["X/USD"],
                      reconciler=reconciler)

    def test_cancel_removes_persisted_order(self):
        rec = self._StubReconciler()
        eng = self._engine(rec)
        eng._active_orders["X/USD"] = {"c1": object()}
        eng._sync_orders("X/USD", 100.0)
        assert eng.broker.cancelled == ["c1"]
        assert rec.removed == ["c1"]

    def test_fill_removes_persisted_order(self):
        from core.strategy import Fill
        rec = self._StubReconciler()
        rec._fills = [Fill(client_id="c1", symbol="X/USD", side="buy",
                           price=100.0, qty=1.0, fee=0.0, ts=time.time())]
        eng = self._engine(rec)
        eng._reconcile_fills()
        assert rec.removed == ["c1"]

    def test_reconciler_remove_order_round_trip(self, tmp_path, monkeypatch):
        import execution.reconciler as recmod
        monkeypatch.setattr(recmod, "_DB_PATH", tmp_path / "t.db")
        r = recmod.Reconciler(lambda since: [])
        r.track_order("c1", "x1", "X/USD", "buy", 100.0, 1.0)
        assert any(o["client_id"] == "c1" for o in r.get_tracked_orders())
        r.remove_order("c1")
        assert r.get_tracked_orders() == []


class TestRefreshRollbackFeatureNames:

    def test_rollback_restores_feature_names(self, monkeypatch):
        """Regression for #119: when a daily refresh trains a worse model on a
        different feature count and is rolled back, _feature_names must be restored
        alongside _clf. Otherwise _clf (34 feats) and _feature_names (16) desync and
        predict()'s feature-count guard trips permanently → silent (hold, 0.0)."""
        import numpy as np
        import pandas as pd
        import ml.trainer as trainer
        from ml.model import TradingModel

        m = TradingModel("TEST/USD")

        class _OldClf:
            classes_ = np.array([0, 1, 2])

            def predict_proba(self, x):
                return np.array([[0.1, 0.2, 0.7]])

        old_clf = _OldClf()
        m._clf = old_clf
        m._n_samples = 500
        m._last_oos_f1 = 0.60
        m._feature_names = [f"f{i}" for i in range(34)]  # good model: 34 features

        # Simulate train(): produces a *worse* model trained on only 16 features,
        # exactly as the 34→16 fallback path would when it fires for the whole window.
        def fake_train(X, y):
            m._clf = object()
            m._n_samples = len(X)
            m._last_oos_f1 = 0.40  # worse by >0.05 → triggers rollback
            m._feature_names = [f"f{i}" for i in range(16)]

        monkeypatch.setattr(m, "train", fake_train)
        monkeypatch.setattr(m, "_save", lambda: None)
        monkeypatch.setattr(TradingModel, "MIN_SAMPLES", 5)
        monkeypatch.setattr(trainer, "_extract_training_features",
                            lambda df, window, btc_corr=0.0: np.zeros(34, np.float32))
        monkeypatch.setattr(trainer, "_get_atr_pct", lambda df, i: 0.01)
        monkeypatch.setattr(trainer, "_compute_label_triple_barrier",
                            lambda df, i, atr_pct: i % 3)

        n = 80
        df = pd.DataFrame({"close": np.linspace(100, 110, n)})
        trainer.refresh_from_recent_history("TEST/USD", df, store=None, model=m)

        # Rollback must restore BOTH the old classifier and its feature names.
        assert m._clf is old_clf
        assert len(m._feature_names) == 34
        # And predict() with a real 34-feature vector must not hit the mismatch guard.
        label, conf = m.predict(np.zeros(34, dtype=np.float32))
        assert label == 2 and conf == pytest.approx(0.7)


# ── optimize.py ready-for-live drawdown gate ─────────────────────────────────

class TestReadyForLiveDrawdown:

    def test_equity_max_drawdown_reads_capital_column(self):
        """Regression for #102: the equity table's value column is `capital`, not
        `equity`. The drawdown helper must read it and compute a real drawdown."""
        import sqlite3
        from scripts.optimize import equity_max_drawdown

        con = sqlite3.connect(":memory:")
        # Mirror dashboard/db.py equity schema exactly.
        con.execute("CREATE TABLE equity (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "timestamp TEXT, capital REAL)")
        curve = [1000, 1100, 1200, 900, 950, 1000, 1050]  # peak 1200 → trough 900 = -25%
        for i, cap in enumerate(curve):
            con.execute("INSERT INTO equity (timestamp, capital) VALUES (?, ?)",
                        (f"2026-07-0{i+1}T00:00:00", cap))
        con.commit()

        dd = equity_max_drawdown(con)
        assert dd is not None, "drawdown must be computable from the `capital` column"
        assert dd == pytest.approx((900 - 1200) / 1200)  # -0.25

    def test_equity_max_drawdown_none_when_short(self):
        import sqlite3
        from scripts.optimize import equity_max_drawdown
        con = sqlite3.connect(":memory:")
        con.execute("CREATE TABLE equity (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "timestamp TEXT, capital REAL)")
        con.execute("INSERT INTO equity (timestamp, capital) VALUES ('2026-07-01', 1000)")
        con.commit()
        assert equity_max_drawdown(con) is None


# ── sweep.py CLI ──────────────────────────────────────────────────────────────

class TestSweepCLI:

    def test_parser_accepts_symbol(self):
        """Regression for #101: nightly_tune passes --symbol; sweep.py's parser must
        accept it (previously it aborted with SystemExit(2), killing every nightly
        sweep)."""
        from scripts.sweep import build_parser
        args = build_parser().parse_args(
            ["--symbol", "SOL/USD", "--days", "180", "--train-days", "120", "--jobs", "4"])
        assert args.symbol == "SOL/USD"
        assert args.days == 180 and args.train_days == 120 and args.jobs == 4

    def test_parser_symbol_optional(self):
        from scripts.sweep import build_parser
        args = build_parser().parse_args([])
        assert args.symbol is None

    def test_run_sweep_disables_live_ml(self, monkeypatch):
        """Regression for #130: cmd_run_sweep is an offline backtest and must build
        GridStrategy with ml_enabled=False — otherwise the ML predict path makes live
        Kraken fetches and paid LightGBM/Claude calls during the sweep."""
        import data_fetcher
        import backtest.engine as bt
        from scripts import optimize

        df = _make_df(120)
        monkeypatch.setattr(data_fetcher, "fetch_ohlcv", lambda *a, **k: df)

        captured = {}

        def fake_run_backtest(strategy, *a, **k):
            captured["ml_enabled"] = strategy._ml_enabled
            return {"total_pnl": 0.0}

        monkeypatch.setattr(bt, "run_backtest", fake_run_backtest)
        optimize.cmd_run_sweep("SOL/USD")
        assert captured["ml_enabled"] is False


# ── MLPredictor error path (#117) ────────────────────────────────────────────

class TestPredictErrorPathClearsScore:
    """Regression for #117: a failed predict() must expire the cached score,
    not leave a stale conviction that adaptive/directional sizing reads."""

    def test_failed_predict_resets_stale_score(self, tmp_path, monkeypatch):
        import ml.data_store as ds
        monkeypatch.setattr(ds, "DB_PATH", tmp_path / "ml.db")   # no repo side effects
        from ml.predictor import MLPredictor

        def boom(*a, **k):
            raise RuntimeError("fetch down")

        p = MLPredictor(fetch_ohlcv_fn=boom)
        p._last_scores["SOL/USD"] = 0.9          # last successful strong-up conviction
        result = p.predict("SOL/USD")            # now fails → except path

        assert result == "neutral"
        assert p.get_score("SOL/USD") == 0.0     # expired, not frozen at 0.9


# ── Directional confidence gating ─────────────────────────────────────────────

class TestDirectionalConfidence:

    def test_hold_confidence_zeroed_for_direction(self):
        """Regression for #103: a confident LightGBM 'hold' must not contribute to
        directional confidence."""
        from ml.predictor import directional_lgbm_conf
        assert directional_lgbm_conf(1, 0.9) == 0.0   # hold → no directional confidence
        assert directional_lgbm_conf(2, 0.9) == 0.9   # buy  → unchanged
        assert directional_lgbm_conf(0, 0.9) == 0.9   # sell → unchanged

    def test_confident_hold_plus_llm_up_stays_below_gate(self):
        """End-to-end numeric check from the issue: LGBM confidently 'hold' (score 0,
        conf 0.9) + LLM 'up' (conf 0.65) must NOT clear MIN_CONFIDENCE once the hold
        confidence is zeroed."""
        from ml.predictor import directional_lgbm_conf, MIN_CONFIDENCE
        from ml.llm_analyst import blend_scores

        lgbm_conf = 0.9
        lgbm_score = 0.0  # hold
        llm_result = {"score": 0.65, "confidence": 0.65}

        # Buggy path (full hold confidence) would clear the gate:
        _, buggy_conf = blend_scores(lgbm_score, lgbm_conf, llm_result)
        assert buggy_conf >= MIN_CONFIDENCE

        # Fixed path (direction-aware confidence) stays below the gate:
        conf_dir = directional_lgbm_conf(1, lgbm_conf)
        _, fixed_conf = blend_scores(lgbm_score, conf_dir, llm_result)
        assert fixed_conf < MIN_CONFIDENCE


# ── Backtest Metrics ─────────────────────────────────────────────────────────

class TestBacktestMetrics:

    def test_sharpe_positive_returns(self):
        from backtest.metrics import sharpe
        import numpy as np
        np.random.seed(1)
        returns = list(0.01 + np.random.normal(0, 0.002, 100))
        s = sharpe(returns)
        assert s > 0

    def test_max_drawdown_negative(self):
        from backtest.metrics import max_drawdown
        equity = [1000, 1100, 900, 950, 1050]
        dd = max_drawdown(equity)
        assert dd < 0

    def test_hit_rate_correct(self):
        from backtest.metrics import hit_rate
        pnls = [10, -5, 8, -3, 12]
        assert hit_rate(pnls) == pytest.approx(0.6)

    def test_profit_factor(self):
        from backtest.metrics import profit_factor
        pnls = [10, -5, 8, -4]
        pf = profit_factor(pnls)
        assert pf == pytest.approx(18 / 9, rel=0.01)

    def test_summary_returns_dict(self):
        from backtest.metrics import summary
        pnls = [5, -2, 8, -3, 6]
        equity = [1000, 1005, 1003, 1011, 1008, 1014]
        result = summary(pnls, equity, days=30)
        assert "sharpe" in result
        assert "max_drawdown_pct" in result
        assert "hit_rate_pct" in result
        assert "profit_factor" in result

    def test_summary_pnls_longer_than_equity_no_indexerror(self):
        """Regression for #131: pnls is filled per SL/TP tick AND per fill while
        equity_curve grows one entry per candle, so len(pnls) can exceed
        len(equity_curve). summary() must not raise IndexError (which the sweep
        silently swallows as {"error": ...} → selection bias)."""
        from backtest.metrics import summary
        pnls = [1.0, -0.5, 2.0, -1.0, 0.7, 3.0, -2.0]   # 7 realized pnls
        equity = [1000, 1002, 1001]                      # only 3 candle snapshots
        assert len(pnls) > len(equity)
        result = summary(pnls, equity, days=10)          # must not raise
        assert result["n_trades"] == 7
        # Returns normalized by starting capital (1000), not a per-candle equity.
        assert result["sharpe"] == pytest.approx(
            summary([p * 2 for p in pnls], [2000, 2004, 2002], days=10)["sharpe"])


# ── Risk Manager ─────────────────────────────────────────────────────────────

class TestRiskManager:

    def _make_rm(self):
        from risk.correlation import CorrelationTracker
        from risk.manager import RiskManager
        return RiskManager(CorrelationTracker())

    def test_can_open_basic(self):
        from core.context import MarketContext
        rm = self._make_rm()
        ctx = MarketContext()
        ctx.set_equity(1000.0)
        rm.set_drawdown_baseline(1000.0)
        ok, reason = rm.can_open("SOL/USD", 50.0, ctx)
        assert ok is True

    def test_blocks_on_daily_drawdown(self):
        from core.context import MarketContext
        rm = self._make_rm()
        ctx = MarketContext()
        ctx.set_equity(1000.0)
        # max_daily_drawdown is 0.10 (10%) per config.yaml (raised from 0.03 in dd0531b).
        # Use an 11% loss so the check still triggers regardless of the configured value.
        rm.set_drawdown_baseline(1124.0)  # 1000/1124 − 1 = −11.0% loss → exceeds 10% threshold
        ok, reason = rm.can_open("SOL/USD", 50.0, ctx)
        assert ok is False
        assert "drawdown" in reason

    def test_drawdown_baseline_anchors_once_no_daily_reset(self):
        # #132: the brake is anchored once to the deposit and must NOT re-anchor
        # on later calls (no daily reset). A later, higher equity value passed in
        # must be ignored so an account that grew still measures drawdown from the
        # original deposit.
        rm = self._make_rm()
        rm.set_drawdown_baseline(1000.0)   # deposit baseline
        rm.set_drawdown_baseline(1200.0)   # later mid-session equity → ignored
        # 1050 is +5% vs deposit(1000) → OK, even though it is −12.5% vs 1200.
        assert rm.drawdown_ok(1050.0) is True
        # 850 is −15% vs the deposit(1000) baseline → tripped.
        assert rm.drawdown_ok(850.0) is False
        # Zero/negative equity never re-anchors the baseline either.
        rm2 = self._make_rm()
        rm2.set_drawdown_baseline(0.0)
        assert rm2.drawdown_ok(500.0) is True  # no baseline yet → permissive
        rm2.set_drawdown_baseline(1000.0)
        assert rm2.drawdown_ok(850.0) is False

    def test_blocks_on_btc_crash(self):
        from core.context import MarketContext, BTCContext
        rm = self._make_rm()
        ctx = MarketContext()
        ctx.set_equity(1000.0)
        rm.set_drawdown_baseline(1000.0)
        ctx.set_btc(BTCContext(
            trend="down", return_1h=-0.05, return_4h=-0.10,
            return_24h=-0.15, realized_vol_7d=0.8, dominance=0.5
        ))
        ok, reason = rm.can_open("SOL/USD", 50.0, ctx)
        assert ok is False
        assert "btc_crash" in reason


# ── GridStrategy ─────────────────────────────────────────────────────────────

class TestGridStrategy:

    def _strategy(self):
        from strategies.grid import GridStrategy
        return GridStrategy([{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}])

    def test_setup_grid_creates_orders(self):
        from core.context import MarketContext
        strategy = self._strategy()
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        strategy.setup_grid("SOL/USD", 100.0, ctx)
        state = strategy.get_state("SOL/USD")
        # ranging regime → 14 levels; grid_lines and orders are non-empty
        assert len(state.orders) > 0
        assert len(state.grid_lines) > 0

    def test_buy_fills_creates_sell(self):
        from core.context import MarketContext
        from core.strategy import Fill
        strategy = self._strategy()
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        state = strategy.get_state("SOL/USD")
        state.with_position = True  # mirror _refresh_prediction: buys only seed when with_position
        strategy.setup_grid("SOL/USD", 100.0, ctx)

        buy_orders = [(cid, o) for cid, o in state.orders.items() if o["side"] == "buy"]
        assert len(buy_orders) > 0
        cid, order = buy_orders[0]

        sells_before = sum(1 for o in state.orders.values() if o["side"] == "sell")
        fill = Fill(client_id=cid, symbol="SOL/USD", side="buy",
                    price=order["price"], qty=order["qty"], fee=0.0, ts=time.time())
        strategy.on_fill(fill, ctx)

        sells_after = sum(1 for o in state.orders.values() if o["side"] == "sell")
        assert sells_after > sells_before

    def test_nongrid_buy_fill_still_gets_sell_and_sl(self):
        """Regression for #127: a buy that fills at a price NOT on the current grid
        (neutral-replenish of a pre-seed sell, or a buy resting from before a
        rebuild) must still get a take-profit sell carrying an sl_price — otherwise
        the position is unsellable and unstopped with unbounded downside."""
        from core.context import MarketContext
        from core.strategy import Fill
        strategy = self._strategy()
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        strategy.setup_grid("SOL/USD", 100.0, ctx)
        state = strategy.get_state("SOL/USD")

        non_grid = state.grid_lines[0] - 0.01  # below every grid line → idx is None
        assert non_grid not in state.grid_lines
        cid = str(uuid.uuid4())
        state.orders[cid] = {"side": "buy", "price": non_grid, "qty": 1.0, "filled": False}
        sells_before = sum(1 for o in state.orders.values() if o["side"] == "sell")

        strategy.on_fill(Fill(client_id=cid, symbol="SOL/USD", side="buy",
                              price=non_grid, qty=1.0, fee=0.0, ts=time.time()), ctx)

        new_sells = [o for o in state.orders.values()
                     if o["side"] == "sell" and o.get("bought_at") == non_grid]
        assert len(new_sells) == 1, "non-grid buy must still create a sell"
        s = new_sells[0]
        assert "sl_price" in s and s["sl_price"] < non_grid  # stop below entry
        assert s["price"] > non_grid                          # TP above entry
        assert sum(1 for o in state.orders.values() if o["side"] == "sell") == sells_before + 1

    def test_top_level_buy_fill_still_gets_sell_and_sl(self):
        """Regression for #127: a buy at the very top grid line (idx == len-1) also
        used to skip sell/SL creation. It must get a synthetic TP + SL now too."""
        from core.context import MarketContext
        from core.strategy import Fill
        strategy = self._strategy()
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        strategy.setup_grid("SOL/USD", 100.0, ctx)
        state = strategy.get_state("SOL/USD")

        top = state.grid_lines[-1]
        cid = str(uuid.uuid4())
        state.orders[cid] = {"side": "buy", "price": top, "qty": 1.0, "filled": False}
        strategy.on_fill(Fill(client_id=cid, symbol="SOL/USD", side="buy",
                              price=top, qty=1.0, fee=0.0, ts=time.time()), ctx)

        new_sells = [o for o in state.orders.values()
                     if o["side"] == "sell" and o.get("bought_at") == top]
        assert len(new_sells) == 1
        assert new_sells[0]["price"] > top
        assert new_sells[0]["sl_price"] < top


    def test_sell_only_latch_blocks_buys_and_survives_refresh(self):
        """Regression for #115: after a graceful wait_fills stop, sell_only must
        block all buys AND survive _refresh_prediction — an "up"/"neutral" prediction
        must not re-arm with_position and let the bot keep buying."""
        from core.context import MarketContext
        from strategies.grid import GridStrategy
        # ml_enabled=False → deterministic rule-based prediction, no network.
        strategy = GridStrategy([{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}],
                                ml_enabled=False)
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        state = strategy.get_state("SOL/USD")

        # A strong uptrend df → the rule-based prediction (ML disabled) yields "up".
        n = 60
        close = pd.Series(np.linspace(100, 140, n))
        df = pd.DataFrame({
            "open": close * 0.999, "high": close * 1.002,
            "low": close * 0.998, "close": close,
            "volume": np.full(n, 1000.0),
        })

        # Sanity: without the latch, an uptrend arms buys.
        strategy._refresh_prediction("SOL/USD", df, ctx)
        assert state.with_position is True
        assert strategy._buys_allowed(state) is True

        # Latch the graceful stop, then let a fresh uptrend prediction run.
        state.sell_only = True
        strategy._refresh_prediction("SOL/USD", df, ctx)
        assert state.with_position is False           # not re-armed
        assert strategy._buys_allowed(state) is False  # buys stay blocked

    def test_per_position_sl_fires_on_tick(self):
        from core.context import MarketContext
        strategy = self._strategy()
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        strategy.setup_grid("SOL/USD", 100.0, ctx)
        state = strategy.get_state("SOL/USD")

        cid = str(uuid.uuid4())
        state.orders[cid] = {
            "side": "sell", "price": 105.0, "qty": 1.0,
            "filled": False, "bought_at": 100.0,
            "sl_price": 95.0, "trailing_activated": False,
        }

        strategy.on_tick("SOL/USD", 94.0, ctx)
        # After SL fires, order is removed from state.orders (cleanup behavior)
        assert cid not in state.orders, "SL order should be removed after firing"
        assert state.total_profit < 0  # SL → loss

    def test_trailing_stop_activates_at_breakeven(self):
        from core.context import MarketContext
        strategy = self._strategy()
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        state = strategy.get_state("SOL/USD")
        state._atr = 2.0  # ATR = $2

        cid = str(uuid.uuid4())
        state.orders[cid] = {
            "side": "sell", "price": 110.0, "qty": 1.0,
            "filled": False, "bought_at": 100.0,
            "sl_price": 96.0, "trailing_activated": False,
        }

        # Price moves up 1×ATR above entry (buy=100, ATR=2 → 102)
        strategy._update_trailing_stops("SOL/USD", 102.5, state)
        assert state.orders[cid]["trailing_activated"] is True
        assert state.orders[cid]["sl_price"] >= 100.0  # SL moved to break-even

    def test_compounding_increases_investment(self):
        from core.context import MarketContext
        strategy = self._strategy()
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        state = strategy.get_state("SOL/USD")
        state.total_profit = 10.0
        state.trade_count = 3  # COMPOUND_EVERY_TRADES = 3
        initial = state.investment
        strategy._maybe_compound(100.0, state)
        assert state.investment > initial


# ── GridParams ────────────────────────────────────────────────────────────────

class TestGridParams:

    def test_defaults_match_legacy_behaviour(self):
        from strategies.grid_params import GridParams
        p = GridParams()
        # sl_mode default is "floor" (cascade-safe); per_position still available
        assert p.sl_mode == "floor"
        assert p.per_pos_sl_step_mult == 1.5
        assert p.per_pos_sl_min_pct == 0.008
        assert p.per_pos_sl_max_pct == 0.04   # hard-cap: no SL wider than 4%
        assert p.momentum_hold_score == 0.35
        assert p.momentum_hold_max == 2
        assert p.regime_levels == {"ranging": 14, "trending": 6, "volatile": 20}
        assert p.trend_filter_enabled is True  # sweep winner 2026-06-12
        assert p.min_step_pct == 0.0
        assert p.directional_enabled is True

    def test_from_dict_roundtrip(self):
        from strategies.grid_params import GridParams
        p = GridParams.from_dict({"sl_mode": "floor", "floor_sl_atr_mult": 1.5,
                                  "levels_by_regime": {"ranging": 10, "trending": 5}})
        d = p.to_dict()
        assert d["sl_mode"] == "floor"
        assert d["floor_sl_atr_mult"] == 1.5
        assert d["levels_by_regime"] == {"ranging": 10, "trending": 5}
        assert GridParams.from_dict(d) == p

    def test_from_dict_ignores_unknown_keys(self):
        from strategies.grid_params import GridParams
        p = GridParams.from_dict({"sl_mode": "floor", "nonsense_key": 42})
        assert p.sl_mode == "floor"


# ── Floor-SL ──────────────────────────────────────────────────────────────────

class TestFloorSL:

    def _strategy(self, **overrides):
        from strategies.grid import GridStrategy
        from strategies.grid_params import GridParams
        params = GridParams.from_dict({"sl_mode": "floor", "leverage": 1.0, **overrides})
        return GridStrategy([{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}],
                            ml_enabled=False, params=params)

    def _setup(self, strategy, price=100.0, atr=2.0):
        from core.context import MarketContext
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        state = strategy.get_state("SOL/USD")
        state._atr = atr
        state.with_position = True  # mirror _refresh_prediction: buys only seed when with_position
        strategy.setup_grid("SOL/USD", price, ctx)
        return ctx, state

    def test_directional_sl_fires_on_safety_tick(self):
        """Regression for #104: an open directional position must be stopped out via
        on_tick_safety (the freeze/emergency safety path), not only via on_tick."""
        strategy = self._strategy(floor_sl_atr_mult=1.0)
        ctx, state = self._setup(strategy)
        state._directional = {
            "entry": 100.0, "qty": 1.0, "usdt": 20.0,
            "tp": 110.0, "sl": 97.0, "entry_ts": time.time(),
        }
        # Price breaks the directional SL. on_tick_safety is what runs during a
        # daily-drawdown freeze; the directional must still exit.
        strategy.on_tick_safety("SOL/USD", 96.0, ctx)
        assert state._directional == {}, "directional SL must fire on the safety tick"

    def test_buy_fill_uses_floor_sl(self):
        from core.strategy import Fill
        strategy = self._strategy(floor_sl_atr_mult=1.0)
        ctx, state = self._setup(strategy)
        assert state.floor_sl > 0
        assert state.floor_sl == pytest.approx(state.grid_lower - 2.0)

        cid, order = [(c, o) for c, o in state.orders.items() if o["side"] == "buy"][0]
        fill = Fill(client_id=cid, symbol="SOL/USD", side="buy",
                    price=order["price"], qty=order["qty"], fee=0.0, ts=time.time())
        strategy.on_fill(fill, ctx)
        sells = [o for o in state.orders.values()
                 if o["side"] == "sell" and "sl_price" in o and not o.get("pre_seeded")]
        assert sells and all(o["sl_price"] == pytest.approx(state.floor_sl) for o in sells)

    def test_no_stop_inside_grid(self):
        from core.strategy import Fill
        strategy = self._strategy(floor_sl_atr_mult=1.0, momentum_hold_max=0)
        ctx, state = self._setup(strategy)
        cid, order = [(c, o) for c, o in state.orders.items() if o["side"] == "buy"][0]
        fill = Fill(client_id=cid, symbol="SOL/USD", side="buy",
                    price=order["price"], qty=order["qty"], fee=0.0, ts=time.time())
        strategy.on_fill(fill, ctx)
        profit_before = state.total_profit
        # Price at the lowest grid line: inside the grid → no stop may fire
        strategy.on_tick("SOL/USD", min(state.grid_lines), ctx)
        assert state.total_profit == profit_before

    def test_floor_break_flushes_all_positions(self):
        from core.strategy import Fill
        strategy = self._strategy(floor_sl_atr_mult=1.0, momentum_hold_max=0)
        ctx, state = self._setup(strategy)
        buys = [(c, o) for c, o in state.orders.items() if o["side"] == "buy"][:2]
        for cid, order in buys:
            fill = Fill(client_id=cid, symbol="SOL/USD", side="buy",
                        price=order["price"], qty=order["qty"], fee=0.0, ts=time.time())
            strategy.on_fill(fill, ctx)
        open_pos = [o for o in state.orders.values()
                    if o["side"] == "sell" and "sl_price" in o and not o.get("pre_seeded")]
        assert len(open_pos) == 2

        strategy.on_tick("SOL/USD", state.floor_sl - 0.01, ctx)
        remaining = [o for o in state.orders.values()
                     if o["side"] == "sell" and "sl_price" in o and not o.get("pre_seeded")]
        assert remaining == []
        assert state.total_profit < 0

    def test_rebuild_never_lowers_sl(self):
        from core.strategy import Fill
        strategy = self._strategy(floor_sl_atr_mult=1.0)
        ctx, state = self._setup(strategy)
        cid, order = [(c, o) for c, o in state.orders.items() if o["side"] == "buy"][0]
        fill = Fill(client_id=cid, symbol="SOL/USD", side="buy",
                    price=order["price"], qty=order["qty"], fee=0.0, ts=time.time())
        strategy.on_fill(fill, ctx)
        pos = [o for o in state.orders.values()
               if o["side"] == "sell" and "sl_price" in o and not o.get("pre_seeded")][0]
        sl_before = pos["sl_price"]

        # Rebuild far lower → new floor far below; existing SL must NOT drop
        strategy.setup_grid("SOL/USD", 80.0, ctx)
        assert pos["sl_price"] >= sl_before


# ── Trend filter ──────────────────────────────────────────────────────────────

class TestTrendFilter:

    def _strategy(self):
        from strategies.grid import GridStrategy
        from strategies.grid_params import GridParams
        params = GridParams.from_dict({"trend_filter_enabled": True, "leverage": 1.0})
        return GridStrategy([{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}],
                            ml_enabled=False, params=params)

    def _df(self, closes):
        closes = np.asarray(closes, dtype=float)
        idx = pd.date_range("2026-01-01", periods=len(closes), freq="h", tz="UTC")
        return pd.DataFrame({
            "open": closes, "high": closes + 0.5, "low": closes - 0.5,
            "close": closes, "volume": 1000.0,
        }, index=idx)

    def test_downtrend_sets_flag_and_blocks_buys(self):
        from core.context import MarketContext
        strategy = self._strategy()
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        state = strategy.get_state("SOL/USD")

        down = self._df(np.linspace(120, 80, 200))
        strategy._update_trend_filter(state, down)
        assert state._hard_trend_down is True
        assert strategy._buys_allowed(state) is False

        state.with_position = True
        strategy.setup_grid("SOL/USD", 80.0, ctx)
        buys = [o for o in strategy.desired_orders("SOL/USD", 80.0, ctx) if o.side == "buy"]
        assert buys == []

    def test_hysteresis_needs_two_clear_candles(self):
        from core.context import MarketContext
        strategy = self._strategy()
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        state = strategy.get_state("SOL/USD")
        state._hard_trend_down = True

        up = self._df(np.linspace(80, 120, 200))
        strategy._update_trend_filter(state, up)
        assert state._hard_trend_down is True  # 1st clear candle: still paused
        strategy._update_trend_filter(state, up)
        assert state._hard_trend_down is False  # 2nd clear candle: resumed


# ── Fee-aware min step ────────────────────────────────────────────────────────

class TestMinStep:

    def test_levels_capped_to_min_step(self):
        from core.context import MarketContext
        from strategies.grid import GridStrategy
        from strategies.grid_params import GridParams
        params = GridParams.from_dict({"min_step_pct": 0.01, "leverage": 1.0})
        strategy = GridStrategy([{"symbol": "SOL/USD", "investment": 100.0, "levels": 20}],
                                ml_enabled=False, params=params)
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        state = strategy.get_state("SOL/USD")
        state._atr = 1.0  # tight volatility → tight range → would violate min step

        strategy.setup_grid("SOL/USD", 100.0, ctx)
        step_pct = (state.grid_lines[1] - state.grid_lines[0]) / 100.0
        assert step_pct >= 0.0099


# ── Backtest equity fidelity ──────────────────────────────────────────────────

class TestBacktestEquity:

    def test_unrealized_losses_visible_in_equity_curve(self):
        """Open underwater positions must drag the equity curve down even
        without realized losses (regression for realized-only equity bug)."""
        from backtest.engine import run_backtest
        from strategies.grid import GridStrategy
        from strategies.grid_params import GridParams

        # fast drop below grid floor (~91.6): fills resting buys before the
        # downtrend buy-pause can cancel them; price held below → losses stay unrealized.
        n_flat, n_drop, n_tail = 70, 4, 46
        closes = np.concatenate([
            100 + 0.3 * np.sin(np.arange(n_flat)),
            np.linspace(100, 88, n_drop),
            88 + 0.3 * np.sin(np.arange(n_tail)),
        ])
        idx = pd.date_range("2026-01-01", periods=len(closes), freq="h", tz="UTC")
        df = pd.DataFrame({
            "open": closes, "high": closes + 0.5, "low": closes - 0.5,
            "close": closes, "volume": 1000.0,
        }, index=idx)

        # Deep floor → nothing stops out → losses stay unrealized.
        # Trend filter off: it would block the buys this test depends on.
        params = GridParams.from_dict({
            "sl_mode": "floor", "floor_sl_atr_mult": 50.0,
            "directional_enabled": False, "leverage": 1.0,
            "trend_filter_enabled": False,
        })
        strategy = GridStrategy([{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}],
                                ml_enabled=False, params=params)
        metrics = run_backtest(strategy, df, "SOL/USD", initial_balance=100.0)
        assert min(metrics["equity_curve"]) < 100.0


# ── Hot-path retrain rollback (#35) ───────────────────────────────────────────

class TestHotPathRetrainRollback:
    """The per-tick online retrain must not replace a good model with a
    materially worse one — same rollback guard the daily refresh already has. #35.
    """

    def _make(self, new_f1):
        import threading
        from ml.trainer import ModelTrainer, RETRAIN_EVERY_N

        class _FakeStore:
            def count_new_labeled_since(self, ts):
                return RETRAIN_EVERY_N
            def get_labeled(self, symbol):
                return (np.zeros((120, 34), dtype=np.float32),
                        np.zeros(120, dtype=np.int32))

        class _FakeModel:
            MIN_SAMPLES = 100
            def __init__(self):
                self._last_oos_f1 = 0.55
                self._clf = "OLD_CLF"
                self._n_samples = 200
                self._lock = threading.Lock()
                self.saves = 0
            def train(self, X, y):  # simulate a successful (gate-passing) retrain
                self._clf = "NEW_CLF"
                self._n_samples = len(X)
                self._last_oos_f1 = new_f1
                self.saves += 1
            def _save(self):
                self.saves += 1

        model = _FakeModel()
        return ModelTrainer(_FakeStore(), {"SOL/USD": model}), model

    def test_worse_model_rolled_back(self):
        trainer, model = self._make(new_f1=0.34)  # 0.55 → 0.34, drop > 0.05
        trainer._maybe_retrain("SOL/USD")
        assert model._clf == "OLD_CLF"        # restored
        assert model._last_oos_f1 == 0.55
        assert model._n_samples == 200

    def test_similar_model_kept(self):
        trainer, model = self._make(new_f1=0.53)  # within 0.05 → keep
        trainer._maybe_retrain("SOL/USD")
        assert model._clf == "NEW_CLF"
        assert model._last_oos_f1 == 0.53
# ── Trainer 34-feature contract (#55) ─────────────────────────────────────────

class TestTrainerFeatureContract:
    """_extract_training_features must return None on failure (caller skips the
    sample) rather than silently falling back to a 16-feature vector. #55.
    """

    def test_returns_none_on_extraction_failure(self, monkeypatch):
        import ml.trainer as trainer
        def _boom(*a, **k):
            raise ValueError("simulated 34-feature extraction failure")
        monkeypatch.setattr(trainer, "extract_all_features", _boom)
        df = _make_df(60)
        out = trainer._extract_training_features(df, df, btc_corr=0.0)
        assert out is None  # NOT a 16-element fallback vector

    def test_returns_34_vector_on_success(self):
        import ml.trainer as trainer
        df = _make_df(120)
        out = trainer._extract_training_features(df, df, btc_corr=0.3)
        assert out is not None
        assert out.shape == (34,)
# ── Trainer stale-candle labeling (#91) ───────────────────────────────────────

class TestTrainerStaleCandleLabel:
    """A sample whose timestamp predates the labeling window must NOT be labeled
    from the clamped first candle (get_indexer nearest never returns -1). #91.
    """

    def _trainer(self, monkeypatch, tmp_path):
        import ml.data_store as data_store
        monkeypatch.setattr(data_store, "DB_PATH", tmp_path / "ml_training.db")
        from ml.data_store import MLDataStore
        from ml.trainer import ModelTrainer
        store = MLDataStore()
        return ModelTrainer(store, {}), store

    def test_out_of_window_sample_not_labeled(self, monkeypatch, tmp_path):
        trainer, store = self._trainer(monkeypatch, tmp_path)
        df = _make_df(60)  # hourly candles starting 2024-01-01
        df.index = df.index.tz_localize("UTC")  # match production UTC-aware index
        sym = "SOL/USD"
        feats = np.zeros(34, dtype=np.float32)

        in_window_ts  = int(df.index[20].timestamp())
        out_window_ts = int(df.index[0].timestamp()) - 100 * 24 * 3600  # 100 days earlier

        store.store(sym, in_window_ts,  feats, 100.0, 1)
        store.store(sym, out_window_ts, feats, 100.0, 1)

        trainer.label_and_maybe_retrain(sym, df)

        # in-window sample got labeled; out-of-window one was skipped (still NULL)
        unlabeled_ts = {ts for _, ts, _ in store.get_unlabeled_before(int(time.time()))}
        assert out_window_ts in unlabeled_ts, "stale sample must NOT be mislabeled"
        assert in_window_ts not in unlabeled_ts, "in-window sample should be labeled"
# ── Correlation-tracker wiring (#43) ──────────────────────────────────────────

class TestCorrelationWiring:
    """RiskManager.can_open() step 5 (over-concentration bucket) was dead because
    CorrelationTracker was never fed. Engine._refresh_correlations() now feeds it.
    """

    def test_refresh_correlations_populates_tracker(self, monkeypatch):
        import types
        import numpy as np
        import pandas as pd
        from core.engine import Engine
        from risk.correlation import CorrelationTracker
        from risk.manager import RiskManager

        idx = pd.date_range("2024-01-01", periods=300, freq="1h")
        rng = np.random.RandomState(0)
        btc = pd.Series(100 + np.cumsum(rng.normal(0, 1.0, 300)), index=idx)
        # SOL strongly driven by BTC → high positive correlation (tiny idiosyncratic noise)
        sol = btc * 0.1 + pd.Series(rng.normal(0, 0.003, 300), index=idx)

        corr = CorrelationTracker()
        risk = RiskManager(corr)

        state = types.SimpleNamespace(_last_df=pd.DataFrame({"close": sol}, index=idx))
        strat = types.SimpleNamespace(_risk=risk, get_state=lambda s: state)
        fake_self = types.SimpleNamespace(strategy=strat, symbols=["SOL/USD"])

        monkeypatch.setattr("market.btc_context.get_btc_close", lambda: btc)

        # Call the unbound method with a fake self (avoids full Engine construction).
        Engine._refresh_correlations(fake_self)

        assert "SOL/USD" in corr._correlations
        assert corr._correlations["SOL/USD"] > 0.9
        assert "SOL/USD" in corr.high_correlation_symbols(0.85)

    def test_refresh_correlations_safe_without_btc_close(self, monkeypatch):
        import types
        from core.engine import Engine
        from risk.correlation import CorrelationTracker
        from risk.manager import RiskManager

        corr = CorrelationTracker()
        risk = RiskManager(corr)
        strat = types.SimpleNamespace(_risk=risk, get_state=lambda s: None)
        fake_self = types.SimpleNamespace(strategy=strat, symbols=["SOL/USD"])
        monkeypatch.setattr("market.btc_context.get_btc_close", lambda: None)

        # No BTC close available → no-op, must not raise.
        Engine._refresh_correlations(fake_self)
        assert corr._correlations == {}
# ── Emergency-stop keeps SL/TP alive (#34) ────────────────────────────────────

class TestEmergencyStopSL:
    """A symbol past its per-coin realized-loss cap must still receive
    on_tick_safety (SL/TP) while new-order paths are blocked — like the freeze.
    Previously the engine `continue`d before on_tick_safety, orphaning positions.
    """

    def _run_one_tick(self, monkeypatch, total_profit):
        import types
        from core.engine import Engine
        from execution.paper import PaperBroker
        from core.context import MarketContext

        calls = {"safety": 0, "on_tick": 0, "sync": 0, "place_new": None}
        state = types.SimpleNamespace(total_profit=total_profit,
                                      investment=100.0, grid_lines=[])

        class FakeStrategy:
            _broker = None
            def get_state(self, s): return state
            def on_tick_safety(self, s, p, ctx): calls["safety"] += 1
            def on_tick(self, s, p, ctx): calls["on_tick"] += 1
            def on_candle(self, s, df, ctx): pass

        broker = PaperBroker(initial_balance=100.0, symbols=["SOL/USD"])
        eng = Engine(FakeStrategy(), broker, ["SOL/USD"],
                     ctx=MarketContext(), initial_capital=100.0)

        # Neutralize everything except the per-symbol gating under test.
        for name in ("_check_dashboard_stop", "_check_daily_drawdown",
                     "_reconcile_fills", "_log_equity"):
            monkeypatch.setattr(eng, name, lambda *a, **k: None)
        monkeypatch.setattr(eng, "_update_dashboard", lambda s, p: None)
        monkeypatch.setattr(eng, "_update_prediction_outcomes", lambda f: None)
        # #222: _sync_orders now runs even for a blocked coin (cancel half); the
        # place_new flag records whether NEW risk was allowed on this call.
        def _spy_sync(s, p, place_new=True):
            calls["sync"] += 1
            calls["place_new"] = place_new
        monkeypatch.setattr(eng, "_sync_orders", _spy_sync)

        import data_fetcher
        monkeypatch.setattr(data_fetcher, "fetch_ticker", lambda s: {"last": 100.0})
        monkeypatch.setattr(data_fetcher, "fetch_ohlcv", lambda *a, **k: None)
        import core.engine as ce
        monkeypatch.setattr(ce.time, "sleep", lambda *a, **k: None)

        eng._loop_count = 1  # off every cadence (recheck/rebuild/btc/funding)
        eng._tick()
        return calls

    def test_healthy_symbol_trades_normally(self, monkeypatch):
        calls = self._run_one_tick(monkeypatch, total_profit=0.0)
        assert (calls["safety"], calls["on_tick"], calls["sync"]) == (1, 1, 1)
        assert calls["place_new"] is True  # healthy coin may open new risk

    def test_emergency_stopped_keeps_sl_blocks_orders(self, monkeypatch):
        # -12% of investment 100 = -12 → total_profit -20 trips the emergency stop.
        calls = self._run_one_tick(monkeypatch, total_profit=-20.0)
        assert calls["safety"] == 1        # SL/TP still runs (was 0 before the fix)
        assert calls["on_tick"] == 0       # no new buys
        # #222: _sync_orders' CANCEL half runs (so stale walls are retracted) …
        assert calls["sync"] == 1
        assert calls["place_new"] is False  # … but the PLACE half opens no new risk


class TestCoinSettingsLiveToggle:
    """#184: the dashboard coin `enabled` toggle must work as a *live* control.
    A coin disabled mid-run is re-read each tick (paper only) and frozen like an
    emergency stop — no new buys (on_tick/_sync_orders), but SL/TP stays live.
    """

    def _run_one_tick(self, monkeypatch, enabled):
        import types
        from core.engine import Engine
        from execution.paper import PaperBroker
        from core.context import MarketContext

        calls = {"safety": 0, "on_tick": 0, "sync": 0, "place_new": None}
        state = types.SimpleNamespace(total_profit=0.0,
                                      investment=100.0, grid_lines=[])

        class FakeStrategy:
            _broker = None
            def get_state(self, s): return state
            def on_tick_safety(self, s, p, ctx): calls["safety"] += 1
            def on_tick(self, s, p, ctx): calls["on_tick"] += 1
            def on_candle(self, s, df, ctx): pass

        broker = PaperBroker(initial_balance=100.0, symbols=["SOL/USD"])
        eng = Engine(FakeStrategy(), broker, ["SOL/USD"],
                     ctx=MarketContext(), initial_capital=100.0)

        for name in ("_check_dashboard_stop", "_check_daily_drawdown",
                     "_reconcile_fills", "_log_equity"):
            monkeypatch.setattr(eng, name, lambda *a, **k: None)
        monkeypatch.setattr(eng, "_update_dashboard", lambda s, p: None)
        monkeypatch.setattr(eng, "_update_prediction_outcomes", lambda f: None)
        # #222: a disabled coin still runs _sync_orders' CANCEL half; place_new
        # records whether the PLACE (new-risk) half was allowed.
        def _spy_sync(s, p, place_new=True):
            calls["sync"] += 1
            calls["place_new"] = place_new
        monkeypatch.setattr(eng, "_sync_orders", _spy_sync)

        # Drive _refresh_coin_settings via the real DB helper (monkeypatched).
        import dashboard.db as ddb
        monkeypatch.setattr(
            ddb, "get_all_coin_settings",
            lambda: [{"symbol": "SOL/USD", "max_investment": 100.0,
                      "enabled": 1 if enabled else 0}],
        )

        import data_fetcher
        monkeypatch.setattr(data_fetcher, "fetch_ticker", lambda s: {"last": 100.0})
        monkeypatch.setattr(data_fetcher, "fetch_ohlcv", lambda *a, **k: None)
        import core.engine as ce
        monkeypatch.setattr(ce.time, "sleep", lambda *a, **k: None)

        eng._loop_count = 1
        eng._tick()
        return calls, eng

    def test_enabled_coin_trades_normally(self, monkeypatch):
        calls, eng = self._run_one_tick(monkeypatch, enabled=True)
        assert (calls["safety"], calls["on_tick"], calls["sync"]) == (1, 1, 1)
        assert calls["place_new"] is True  # enabled coin may open new risk
        assert eng._disabled_coins == set()

    def test_disabled_coin_keeps_sl_blocks_orders(self, monkeypatch):
        calls, eng = self._run_one_tick(monkeypatch, enabled=False)
        assert calls["safety"] == 1        # SL/TP still protects open positions
        assert calls["on_tick"] == 0       # no new buys
        # #222: CANCEL half runs (retracts stale walls) but no new risk is placed.
        assert calls["sync"] == 1
        assert calls["place_new"] is False
        assert eng._disabled_coins == {"SOL/USD"}

    def test_refresh_is_paper_only(self, monkeypatch):
        # Live broker must not be touched (Live-Parität lock, #171) — refresh no-ops.
        from core.engine import Engine
        from core.context import MarketContext

        class FakeLiveBroker:  # not a PaperBroker
            def cancel_all(self, s): pass

        called = {"n": 0}
        import dashboard.db as ddb
        monkeypatch.setattr(ddb, "get_all_coin_settings",
                            lambda: called.__setitem__("n", called["n"] + 1) or [])
        eng = Engine(object(), FakeLiveBroker(), ["SOL/USD"],
                     ctx=MarketContext(), initial_capital=100.0)
        eng._refresh_coin_settings()
        assert called["n"] == 0                 # DB never queried for a live broker
        assert eng._disabled_coins == set()
# ── Engine equity staleness (#89) ─────────────────────────────────────────────

class TestEquityStaleGuard:
    """A permanently-failing ticker must NOT freeze the whole equity curve /
    daily-drawdown brake forever. Brief staleness is skipped (sleep-wake guard,
    #20); persistent staleness falls back to last-good prices (#89).
    """

    class _FakeState:
        def __init__(self):
            # one open (unfilled) grid sell with a known entry → deterministic MTM
            self.orders = {
                "s1": {"side": "sell", "filled": False, "bought_at": 90.0,
                       "qty": 1.0, "leverage": 1.0},
            }
            self.total_profit = 0.0

    class _FakeStrategy:
        def __init__(self, state):
            self._state = state
        def get_state(self, sym):
            return self._state

    class _FakeBroker:
        def get_balance(self, currency="USD"):
            return 500.0

    def _engine(self):
        from core.engine import Engine
        from core.context import MarketContext
        state = self._FakeState()
        eng = Engine(self._FakeStrategy(state), self._FakeBroker(),
                     ["SOL/USD"], ctx=MarketContext())
        # last-good price 100 → MTM = margin(90) + unrealized(1*(100-90)) = 100
        eng._last_prices = {"SOL/USD": 100.0}
        return eng

    def test_brief_staleness_skips_equity_update(self):
        eng = self._engine()
        eng.ctx.set_equity(999.0)  # sentinel
        eng._last_price_ts = {"SOL/USD": time.time() - 10_000}  # very stale
        eng._log_equity()
        # within grace → skipped, sentinel retained (no false drawdown trigger)
        assert eng.ctx.total_equity == 999.0
        assert eng._equity_stale_since is not None

    def test_persistent_staleness_falls_back_to_last_good_price(self):
        from core.engine import STALE_EQUITY_GRACE_SECONDS
        eng = self._engine()
        eng.ctx.set_equity(999.0)  # sentinel
        eng._last_price_ts = {"SOL/USD": time.time() - 10_000}
        # pretend staleness began well before the grace window
        eng._equity_stale_since = time.time() - (STALE_EQUITY_GRACE_SECONDS + 30)
        eng._log_equity()
        # grace expired → equity logged with last-good price: 500 balance + 100 MTM
        assert eng.ctx.total_equity == pytest.approx(600.0)

    def test_fresh_prices_reset_stale_marker(self):
        eng = self._engine()
        eng._equity_stale_since = time.time() - 5.0  # was stale
        eng._last_price_ts = {"SOL/USD": time.time()}  # now fresh
        eng._log_equity()
        assert eng._equity_stale_since is None
        assert eng.ctx.total_equity == pytest.approx(600.0)
# ── cancel_all failure visibility (#61) ───────────────────────────────────────

class TestCancelAllLogging:
    """cancel_all must not silently swallow cancel failures — a stale live order
    left on the book after 'stop' can fill unexpectedly."""

    def test_cancel_all_counts_successes_and_logs_failures(self, caplog):
        import types
        import logging
        from execution.kraken import KrakenBroker

        orders = [types.SimpleNamespace(exchange_order_id=oid) for oid in ("a", "b", "c")]

        def cancel_order(oid):
            if oid == "b":
                raise RuntimeError("network down")  # not a ccxt err → no retry delay
            return {}

        fake = types.SimpleNamespace(
            get_open_orders=lambda s: orders,
            _ex=types.SimpleNamespace(cancel_order=cancel_order),
        )
        with caplog.at_level(logging.WARNING):
            count = KrakenBroker.cancel_all(fake, "SOL/USD")

        assert count == 2  # a and c cancelled; b failed
        msgs = " ".join(r.getMessage() for r in caplog.records)
        assert "cancel failed" in msgs
        assert "still OPEN" in msgs
        assert "b" in msgs
# ── Live reconciler robustness (#76) ──────────────────────────────────────────

class TestReconcileFeeNone:
    """reconcile_fills must not drop the whole batch when one ccxt trade has
    fee=None (the key exists but is None, so t.get('fee', {}) returns None)."""

    def _fills(self, trades):
        import types
        from execution.kraken import KrakenBroker
        fake = types.SimpleNamespace(
            _ex=types.SimpleNamespace(fetch_my_trades=lambda *a, **k: trades),
            _client_to_exchange={},
        )
        return KrakenBroker.reconcile_fills(fake, 0.0)

    def test_fee_none_does_not_drop_batch(self):
        trades = [
            {"order": "o1", "symbol": "SOL/USD", "side": "buy", "price": 100.0,
             "amount": 1.0, "fee": {"cost": 0.16}, "timestamp": 1000, "id": "t1"},
            {"order": "o2", "symbol": "SOL/USD", "side": "sell", "price": 110.0,
             "amount": 1.0, "fee": None, "timestamp": 2000, "id": "t2"},  # fee=None
        ]
        fills = self._fills(trades)
        assert len(fills) == 2
        assert fills[0].fee == pytest.approx(0.16)
        assert fills[1].fee == 0.0

    def test_one_malformed_trade_skipped_others_survive(self):
        trades = [
            {"order": "o1", "symbol": "SOL/USD", "side": "buy", "price": 100.0,
             "amount": 1.0, "fee": {"cost": 0.1}, "timestamp": 1000, "id": "t1"},
            {"order": "bad", "id": "t2"},  # missing price/symbol → skipped
            {"order": "o3", "symbol": "ETH/USD", "side": "sell", "price": 3000.0,
             "amount": 0.5, "fee": {"cost": 2.4}, "timestamp": 3000, "id": "t3"},
        ]
        fills = self._fills(trades)
        assert len(fills) == 2
        assert {f.symbol for f in fills} == {"SOL/USD", "ETH/USD"}
# ── Floor-SL paper credit (#39 double-fee, #57 entry-leverage) ────────────────

class TestFloorSLCredit:
    """The floor-SL exit credits margin+PnL back to the PaperBroker manually
    (the broker never sees the SL fill). Regression guards for two bugs:
      #39 — a full round-trip fee was charged, double-counting the buy fee that
            was already deducted at buy-fill time.
      #57 — the live dashboard leverage was used to return margin instead of the
            leverage the position was entered with → balance drift on lev change.
    """

    def _sl_credit_amount(self, monkeypatch, entry_lev, live_lev):
        from strategies.grid import GridStrategy, _GridState
        from strategies.grid_params import GridParams
        from execution.paper import PaperBroker
        from core.context import MarketContext

        monkeypatch.setenv("GRIDBOT_BACKTEST", "1")  # skip dashboard logging

        strat = GridStrategy(
            [{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}],
            ml_enabled=False,
            params=GridParams(leverage=live_lev),
        )
        broker = PaperBroker(initial_balance=100.0, symbols=["SOL/USD"])
        strat._broker = broker

        state = _GridState("SOL/USD", 100.0, 6, 0.05)
        state.orders["sell1"] = {
            "side": "sell", "price": 110.0, "qty": 1.0, "filled": False,
            "bought_at": 100.0, "sl_price": 99.0, "leverage": entry_lev,
            "momentum_holds": 0,
        }
        strat._states["SOL/USD"] = state

        captured = {}
        orig = broker.sl_credit
        monkeypatch.setattr(
            broker, "sl_credit",
            lambda symbol, amount: (captured.__setitem__("amount", amount),
                                    orig(symbol, amount))[1],
        )

        # price 98 < sl_price 99 → floor-SL fires
        strat._check_position_stops("SOL/USD", 98.0, state, MarketContext())
        return captured.get("amount")

    def test_sl_credit_charges_sell_fee_only(self, monkeypatch):
        from strategies.grid import KRAKEN_FEE
        amount = self._sl_credit_amount(monkeypatch, entry_lev=1.0, live_lev=1.0)
        # margin(100) + pnl(-2) - sell_fee(98 * KRAKEN_FEE); buy fee NOT re-charged
        expected = 100.0 + (98.0 - 100.0) - 98.0 * KRAKEN_FEE
        assert amount == pytest.approx(expected)
        assert amount == pytest.approx(97.8432)

    def test_sl_credit_uses_entry_leverage(self, monkeypatch):
        from strategies.grid import KRAKEN_FEE
        # Entered at lev=2, dashboard later switched to lev=1 → still use lev=2.
        amount = self._sl_credit_amount(monkeypatch, entry_lev=2.0, live_lev=1.0)
        expected = 100.0 / 2.0 + (98.0 - 100.0) - 98.0 * KRAKEN_FEE
        assert amount == pytest.approx(expected)


class TestEmergencySellAllCredit:
    """#180: the graceful "sell_all" paper stop must credit margin+PnL back to
    the broker cash bucket, exactly like the floor-SL path. Before the fix the
    synthetic sell only updated in-memory total_profit and logged the trade;
    the broker sell order was already cancelled, so update_price never credited
    the margin -> the persisted balance leaked margin+unrealized on every stop.
    """

    def _run_emergency(self, monkeypatch, entry_lev, sell_price):
        from core.engine import Engine, KRAKEN_FEE
        from execution.paper import PaperBroker
        from strategies.grid import GridStrategy, _GridState
        from strategies.grid_params import GridParams
        from core.context import MarketContext

        monkeypatch.setenv("GRIDBOT_BACKTEST", "1")  # skip dashboard/notifier

        strat = GridStrategy(
            [{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}],
            ml_enabled=False,
            params=GridParams(leverage=entry_lev),
        )
        broker = PaperBroker(initial_balance=100.0, symbols=["SOL/USD"])
        strat._broker = broker

        state = _GridState("SOL/USD", 100.0, 6, 0.05)
        state.orders["sell1"] = {
            "side": "sell", "price": 110.0, "qty": 1.0, "filled": False,
            "bought_at": 100.0, "sl_price": 99.0, "leverage": entry_lev,
            "momentum_holds": 0, "entry_ts": 0.0,
        }
        strat._states["SOL/USD"] = state

        eng = Engine(strat, broker, ["SOL/USD"],
                     ctx=MarketContext(), initial_capital=100.0)

        import data_fetcher
        monkeypatch.setattr(data_fetcher, "fetch_ticker",
                            lambda s: {"last": sell_price})

        bucket_before = broker._balances["SOL/USD"]
        eng._emergency_sell_all()
        return broker._balances["SOL/USD"], bucket_before, KRAKEN_FEE

    def test_bucket_credited_margin_and_pnl(self, monkeypatch):
        after, before, fee = self._run_emergency(monkeypatch, entry_lev=1.0,
                                                 sell_price=105.0)
        # margin(100) + pnl(+5) - sell_fee(105*fee); buy fee NOT re-charged.
        credit = 100.0 / 1.0 + (105.0 - 100.0) - 105.0 * fee
        assert after == pytest.approx(before + credit)

    def test_bucket_credit_uses_entry_leverage(self, monkeypatch):
        after, before, fee = self._run_emergency(monkeypatch, entry_lev=2.0,
                                                 sell_price=105.0)
        credit = 100.0 / 2.0 + (105.0 - 100.0) - 105.0 * fee
        assert after == pytest.approx(before + credit)


class TestLeverageChangeBetweenPlaceAndFill:
    """#206: the broker deducts a grid buy's margin using the leverage stamped on
    the order at *placement* time (L1). The sell/SL that later returns that margin
    must use the SAME leverage. Before the fix, _handle_buy_fill re-read the live
    dashboard leverage at *fill* time (L2); if the user changed leverage while the
    buy rested as a limit order, L1 != L2 and the paper cash bucket drifted by
    bought_at*qty*(1/L1 - 1/L2) per position — corrupting the deposit-anchored
    drawdown brake. The fix pins the return leverage to the buy's fill.meta.
    """

    def _round_trip(self, monkeypatch, change_lev_to):
        """Full buy→sell round trip through the real broker. Leverage starts at
        1.0; after the buy is placed (but before it fills) it is set to
        `change_lev_to`. Returns the final SOL/USD cash bucket."""
        from strategies.grid import GridStrategy, _GridState
        from strategies.grid_params import GridParams
        from execution.paper import PaperBroker
        from core.context import MarketContext

        monkeypatch.setenv("GRIDBOT_BACKTEST", "1")  # skip dashboard/notifier

        strat = GridStrategy(
            [{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}],
            ml_enabled=False,
            params=GridParams(sl_mode="per_position", leverage=0.0,
                              trend_filter_enabled=False,
                              max_inventory_notional_mult=0.0,
                              min_confidence_to_buy=0.0),
        )
        broker = PaperBroker(initial_balance=100.0, symbols=["SOL/USD"])
        strat._broker = broker
        ctx = MarketContext()

        lev = {"v": 1.0}
        monkeypatch.setattr(strat, "_lev", lambda: lev["v"])

        state = _GridState("SOL/USD", 100.0, 6, 0.05)
        state.grid_lines = [90.0, 100.0, 110.0]
        state.with_position = True
        # A resting grid buy at 100. Deliberately NO "leverage" key here: it is
        # desired_orders that stamps the broker meta leverage from _lev() at emit
        # time, exactly like a live grid buy that predates a rebuild-freeze.
        buy_cid = "buy1"
        state.orders[buy_cid] = {"side": "buy", "price": 100.0, "qty": 0.1,
                                 "filled": False}
        strat._states["SOL/USD"] = state

        # 1. Emit + place the buy while leverage is still L1 = 1.0.
        emitted = strat.desired_orders("SOL/USD", 100.0, ctx)
        buy_order = next(o for o in emitted if o.client_id == buy_cid)
        broker.place_limit(symbol="SOL/USD", side="buy", price=100.0, qty=0.1,
                           client_id=buy_cid, meta=buy_order.meta)

        # 2. User changes the dashboard leverage while the buy rests.
        lev["v"] = change_lev_to

        # 3. Buy fills — broker deducts margin using the PLACEMENT meta (L1=1.0).
        buy_fills = broker.update_price("SOL/USD", 99.0)
        assert len(buy_fills) == 1
        strat.on_fill(buy_fills[0], ctx)

        # 4. Emit + place the sell the buy created, then fill it at 110.
        emitted2 = strat.desired_orders("SOL/USD", 105.0, ctx)
        sell_order = next(o for o in emitted2 if o.side == "sell")
        broker.place_limit(symbol="SOL/USD", side="sell", price=sell_order.price,
                           qty=sell_order.qty, client_id=sell_order.client_id,
                           meta=sell_order.meta)
        sell_fills = broker.update_price("SOL/USD", 111.0)
        assert len(sell_fills) == 1

        return broker._balances["SOL/USD"]

    def test_no_drift_when_leverage_changes_mid_position(self, monkeypatch):
        # A position opened at L1=1.0 must settle identically whether or not the
        # dashboard leverage is later bumped to 3.0 — the change must not touch a
        # position that was already margined at the old leverage.
        stable  = self._round_trip(monkeypatch, change_lev_to=1.0)
        changed = self._round_trip(monkeypatch, change_lev_to=3.0)
        assert changed == pytest.approx(stable)
        # sanity: the round trip actually completed with a profit (sell 110 > buy 100)
        assert stable > 100.0


class TestBuyFillQtyAfterRebuildResize:
    """#208: residual of #206 (point 3). A resting grid buy that survives a grid
    rebuild reuses its client_id but is re-sized to the new lev/investment in
    state.orders, while _sync_orders never re-places an already-active cid — so
    the broker still holds (and fills) the OLD qty. _handle_buy_fill must stamp
    the resulting sell/position with the qty the broker ACTUALLY filled (fill.qty),
    not state.orders[cid]["qty"] (the new rebuild qty). Otherwise the sell returns
    margin for a quantity that was never bought → paper-cash drift feeding the
    deposit-anchored drawdown brake. The fix uses fill.qty.
    """

    def _round_trip(self, monkeypatch, resize_qty_to):
        """Full buy→sell round trip through the real broker. The buy is placed at
        qty 0.1; after placement (but before it fills) state.orders[cid]["qty"] is
        overwritten with `resize_qty_to` — emulating a rebuild that reused the cid
        and re-sized it while the broker order still holds the original 0.1.
        Returns the final SOL/USD cash bucket."""
        from strategies.grid import GridStrategy, _GridState
        from strategies.grid_params import GridParams
        from execution.paper import PaperBroker
        from core.context import MarketContext

        monkeypatch.setenv("GRIDBOT_BACKTEST", "1")  # skip dashboard/notifier

        strat = GridStrategy(
            [{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}],
            ml_enabled=False,
            params=GridParams(sl_mode="per_position", leverage=0.0,
                              trend_filter_enabled=False,
                              max_inventory_notional_mult=0.0,
                              min_confidence_to_buy=0.0),
        )
        broker = PaperBroker(initial_balance=100.0, symbols=["SOL/USD"])
        strat._broker = broker
        ctx = MarketContext()

        # Leverage is constant here — this test isolates the qty path (not #206's
        # leverage path). Kept at 1.0 for both round trips.
        monkeypatch.setattr(strat, "_lev", lambda: 1.0)

        state = _GridState("SOL/USD", 100.0, 6, 0.05)
        state.grid_lines = [90.0, 100.0, 110.0]
        state.with_position = True
        buy_cid = "buy1"
        state.orders[buy_cid] = {"side": "buy", "price": 100.0, "qty": 0.1,
                                 "filled": False, "leverage": 1.0}
        strat._states["SOL/USD"] = state

        # 1. Emit + place the buy at qty 0.1.
        emitted = strat.desired_orders("SOL/USD", 100.0, ctx)
        buy_order = next(o for o in emitted if o.client_id == buy_cid)
        broker.place_limit(symbol="SOL/USD", side="buy", price=100.0, qty=0.1,
                           client_id=buy_cid, meta=buy_order.meta)

        # 2. A rebuild reuses the cid and re-sizes the resting buy in state.orders;
        #    the already-placed broker order is NOT touched (still qty 0.1).
        state.orders[buy_cid]["qty"] = resize_qty_to

        # 3. Buy fills — broker fills its OWN order qty (0.1) and deducts for 0.1.
        buy_fills = broker.update_price("SOL/USD", 99.0)
        assert len(buy_fills) == 1
        assert buy_fills[0].qty == pytest.approx(0.1)
        strat.on_fill(buy_fills[0], ctx)

        # 4. Emit + place the sell the buy created, then fill it at 110.
        emitted2 = strat.desired_orders("SOL/USD", 105.0, ctx)
        sell_order = next(o for o in emitted2 if o.side == "sell")
        broker.place_limit(symbol="SOL/USD", side="sell", price=sell_order.price,
                           qty=sell_order.qty, client_id=sell_order.client_id,
                           meta=sell_order.meta)
        sell_fills = broker.update_price("SOL/USD", 111.0)
        assert len(sell_fills) == 1

        return broker._balances["SOL/USD"]

    def test_no_drift_when_buy_resized_mid_rest(self, monkeypatch):
        # A position the broker actually bought at qty 0.1 must settle identically
        # whether or not state.orders was later re-sized to 0.3 by a rebuild — the
        # re-size must not credit margin for qty that was never purchased.
        stable  = self._round_trip(monkeypatch, resize_qty_to=0.1)
        resized = self._round_trip(monkeypatch, resize_qty_to=0.3)
        assert resized == pytest.approx(stable)
        # sanity: the round trip actually completed with a profit (sell 110 > buy 100)
        assert stable > 100.0


class TestPaperRestartMargin:
    """#183: normal SIGTERM/restart/crash never sells open positions, and only
    the cash balance is persisted (not the positions). Without settling them,
    each restart drops the persisted equity by the full margin + unrealized PnL.
    _mtm_close_paper_positions() must credit exactly margin+unrealized (== the MTM
    _log_equity reports) into the bucket so restart equity == displayed equity.
    """

    def _build_engine(self, entry_lev, last_price):
        from core.engine import Engine
        from execution.paper import PaperBroker
        from strategies.grid import GridStrategy, _GridState
        from strategies.grid_params import GridParams
        from core.context import MarketContext

        strat = GridStrategy(
            [{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}],
            ml_enabled=False,
            params=GridParams(leverage=entry_lev),
        )
        broker = PaperBroker(initial_balance=100.0, symbols=["SOL/USD"])
        strat._broker = broker

        state = _GridState("SOL/USD", 100.0, 6, 0.05)
        state.orders["sell1"] = {
            "side": "sell", "price": 110.0, "qty": 1.0, "filled": False,
            "bought_at": 100.0, "sl_price": 99.0, "leverage": entry_lev,
        }
        strat._states["SOL/USD"] = state

        eng = Engine(strat, broker, ["SOL/USD"],
                     ctx=MarketContext(), initial_capital=100.0)
        eng._last_prices["SOL/USD"] = last_price
        return eng, broker, state

    def test_mtm_close_credits_margin_and_unrealized(self):
        eng, broker, state = self._build_engine(entry_lev=1.0, last_price=105.0)
        before = broker._balances["SOL/USD"]
        eng._mtm_close_paper_positions()
        # margin(100) + unrealized(+5); no synthetic fee -> equity continuity.
        assert broker._balances["SOL/USD"] == pytest.approx(before + 105.0)
        assert "sell1" not in state.orders  # credited exactly once

    def test_mtm_close_uses_entry_leverage(self):
        eng, broker, state = self._build_engine(entry_lev=2.0, last_price=105.0)
        before = broker._balances["SOL/USD"]
        eng._mtm_close_paper_positions()
        # margin(100/2=50) + unrealized(+5)
        assert broker._balances["SOL/USD"] == pytest.approx(before + 55.0)

    def test_mtm_close_skips_without_known_price(self):
        eng, broker, state = self._build_engine(entry_lev=1.0, last_price=105.0)
        eng._last_prices["SOL/USD"] = 0.0  # no known price
        before = broker._balances["SOL/USD"]
        eng._mtm_close_paper_positions()
        assert broker._balances["SOL/USD"] == pytest.approx(before)  # unchanged
        assert "sell1" in state.orders  # left intact, not lost

    def test_mtm_close_ignores_pre_seeded(self):
        eng, broker, state = self._build_engine(entry_lev=1.0, last_price=105.0)
        state.orders["seed1"] = {
            "side": "sell", "price": 120.0, "qty": 1.0, "filled": False,
            "bought_at": 100.0, "pre_seeded": True, "leverage": 1.0,
        }
        before = broker._balances["SOL/USD"]
        eng._mtm_close_paper_positions()
        # only the real position (sell1) is credited; pre-seeded had no margin.
        assert broker._balances["SOL/USD"] == pytest.approx(before + 105.0)
        assert "seed1" in state.orders
# ── Latent correctness traps (#78) ────────────────────────────────────────────

class TestLatentTraps:

    def test_level_allocations_empty_grid_no_zerodiv(self):
        # #78.1: n == 0 must early-out to {}, not fall into `investment / n`.
        from strategies.grid import _calc_level_allocations
        assert _calc_level_allocations([], 100.0, 40.0, 0.5) == {}

    def test_level_allocations_uniform_when_neutral(self):
        # Non-empty grid, neutral score → uniform split (sanity, unchanged path).
        from strategies.grid import _calc_level_allocations
        alloc = _calc_level_allocations([10.0, 11.0], 10.5, 40.0, 0.0)
        assert alloc == {10.0: 20.0, 11.0: 20.0}

    def test_risk_position_size_no_leverage_param(self):
        # #78.2: the misleading (silently-ignored) leverage param is gone.
        import inspect
        from risk.manager import RiskManager
        assert "leverage" not in inspect.signature(RiskManager.position_size).parameters

    def test_data_fetcher_get_balance_defaults_usd(self):
        # #78.3: default currency must be USD (Kraken/USD account), not USDT.
        import inspect
        import data_fetcher
        assert inspect.signature(data_fetcher.get_balance).parameters["currency"].default == "USD"


# ── coin_settings enable/disable + budget cap (#114) ──────────────────────────
class TestCoinSettingsGridConfig:
    """coin_settings must disable coins and cap budget in BOTH paper and live mode
    (the logic is mode-independent)."""

    def test_disabled_coin_is_dropped(self):
        from main import _build_grids_config
        settings = {"ETH/USD": {"enabled": 0, "max_investment": 300.0}}
        active, cfg = _build_grids_config(["SOL/USD", "ETH/USD"], 200.0, settings)
        assert active == ["SOL/USD"]
        assert [c["symbol"] for c in cfg] == ["SOL/USD"]

    def test_missing_row_defaults_enabled(self):
        from main import _build_grids_config
        active, cfg = _build_grids_config(["SOL/USD"], 200.0, {})
        assert active == ["SOL/USD"]
        assert cfg[0]["investment"] == 200.0  # full per_coin bucket

    def test_max_investment_only_reduces(self):
        from main import _build_grids_config
        settings = {
            "SOL/USD": {"enabled": 1, "max_investment": 50.0},    # reduce
            "ETH/USD": {"enabled": 1, "max_investment": 9999.0},  # cannot exceed per_coin
        }
        _, cfg = _build_grids_config(["SOL/USD", "ETH/USD"], 200.0, settings)
        inv = {c["symbol"]: c["investment"] for c in cfg}
        assert inv["SOL/USD"] == 50.0
        assert inv["ETH/USD"] == 200.0

    def test_all_disabled_yields_empty(self):
        from main import _build_grids_config
        settings = {"SOL/USD": {"enabled": 0}, "ETH/USD": {"enabled": 0}}
        active, cfg = _build_grids_config(["SOL/USD", "ETH/USD"], 200.0, settings)
        assert active == [] and cfg == []


# ── Double SL/directional checks per tick (#146) ──────────────────────────────
class TestTickCheckDedup:
    """on_tick_safety + on_tick must not run the SL/directional checks twice in a
    single engine tick, or the momentum_hold_max SL deferral is halved."""

    def _strategy(self, **overrides):
        from strategies.grid import GridStrategy
        from strategies.grid_params import GridParams
        params = GridParams.from_dict({"sl_mode": "floor", "leverage": 1.0, **overrides})
        return GridStrategy([{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}],
                            ml_enabled=False, params=params)

    def _spy(self, strategy):
        calls = {"stops": 0, "dir": 0}
        strategy._check_position_stops = lambda *a, **k: calls.__setitem__("stops", calls["stops"] + 1)
        strategy._check_directional    = lambda *a, **k: calls.__setitem__("dir", calls["dir"] + 1)
        strategy._maybe_open_directional = lambda *a, **k: None
        strategy._check_mtf_entry        = lambda *a, **k: None
        strategy._update_trailing_stops  = lambda *a, **k: None
        return calls

    def test_safety_then_on_tick_runs_checks_once(self):
        from core.context import MarketContext
        strategy = self._strategy()
        strategy.init(["SOL/USD"], MarketContext())
        ctx = MarketContext()
        calls = self._spy(strategy)
        # Mirror the engine's per-tick order: safety first, then on_tick.
        strategy.on_tick_safety("SOL/USD", 100.0, ctx)
        strategy.on_tick("SOL/USD", 100.0, ctx)
        assert calls == {"stops": 1, "dir": 1}

    def test_standalone_on_tick_still_runs_checks(self):
        from core.context import MarketContext
        strategy = self._strategy()
        strategy.init(["SOL/USD"], MarketContext())
        ctx = MarketContext()
        calls = self._spy(strategy)
        # Called without a preceding safety tick → must run the checks itself.
        strategy.on_tick("SOL/USD", 100.0, ctx)
        assert calls == {"stops": 1, "dir": 1}


# ── Confident-neutral LLM must not inflate blended confidence (#129) ───────────
class TestLLMNeutralConfidence:
    def test_neutral_llm_does_not_inflate_confidence(self):
        from ml.llm_analyst import blend_scores, LLM_WEIGHT
        # Confident neutral LLM (passes the gate) must contribute 0 to directional conf.
        neutral = {"direction": "neutral", "confidence": 0.90, "score": 0.0}
        blended, conf = blend_scores(0.50, 0.50, neutral)
        assert conf == (1 - LLM_WEIGHT) * 0.50  # LGBM-only confidence, LLM zeroed
        assert conf < 0.50 + 1e-9

    def test_directional_llm_still_contributes_confidence(self):
        from ml.llm_analyst import blend_scores, LLM_WEIGHT
        up = {"direction": "up", "confidence": 0.90, "score": 0.90}
        blended, conf = blend_scores(0.50, 0.50, up)
        assert conf == (1 - LLM_WEIGHT) * 0.50 + LLM_WEIGHT * 0.90


# ── LLM confidence/score clamping (#151) ──────────────────────────────────────
class TestLLMClamp:
    """A malformed Haiku confidence (e.g. a percentage like 85) must not survive
    into a max-conviction leveraged directional trade."""

    def test_blend_clamps_out_of_range_confidence_and_score(self):
        from ml.llm_analyst import blend_scores
        # LLM returned confidence=85 (percent) and an out-of-range score.
        bad = {"direction": "up", "confidence": 85.0, "score": 85.0}
        blended, conf = blend_scores(0.2, 0.5, bad)
        assert -1.0 <= blended <= 1.0
        assert 0.0 <= conf <= 1.0

    def test_blend_none_result_passes_through(self):
        from ml.llm_analyst import blend_scores
        assert blend_scores(0.3, 0.7, None) == (0.3, 0.7)

    def test_blend_below_gate_returns_lgbm_only(self):
        from ml.llm_analyst import blend_scores, LLM_CONFIDENCE_MIN
        low = {"direction": "up", "confidence": LLM_CONFIDENCE_MIN - 0.1, "score": 1.0}
        assert blend_scores(0.3, 0.7, low) == (0.3, 0.7)


# ── PaperBroker balance provisioning (#149) ───────────────────────────────────
class TestPaperBalanceProvisioning:
    """With per-symbol buckets, the fallback pool must not double-provision the
    account, and get_balance() must include every pool it can move money to."""

    def test_seeded_buckets_do_not_double_provision(self):
        from execution.paper import PaperBroker
        b = PaperBroker(initial_balance=900.0, symbols=["A/USD", "B/USD", "C/USD"])
        # Buckets already sum to 900; the hidden fallback pool must be 0, not 900.
        assert b._balance == 0.0
        assert b.get_balance() == 900.0

    def test_unseeded_symbol_cannot_spend_the_account(self):
        from execution.paper import PaperBroker
        b = PaperBroker(initial_balance=900.0, symbols=["A/USD", "B/USD", "C/USD"])
        # A symbol not in the seed list has no bucket → buy must be rejected by the
        # affordability guard instead of draining a hidden full-account pool.
        b.place_limit("Z/USD", "buy", 100.0, 1.0, client_id="z1")
        fills = b.update_price("Z/USD", 100.0)
        assert fills == []
        assert b.get_balance() == 900.0

    def test_no_symbols_keeps_single_pool(self):
        from execution.paper import PaperBroker
        b = PaperBroker(initial_balance=500.0)
        assert b.get_balance() == 500.0


class TestLeverageEndpointClamp:
    """#150: POST /api/leverage must echo the *persisted* (clamped) leverage, not
    the raw request value, and return 400 (not 500) on non-numeric input."""

    def _client(self, monkeypatch, tmp_path):
        import dashboard.db as ddb
        monkeypatch.setattr(ddb, "DB_PATH", tmp_path / "trades.db")
        import dashboard.app as dapp
        dapp.app.config["TESTING"] = True
        return dapp.app.test_client()

    def test_over_max_leverage_echoes_clamped_value(self, monkeypatch, tmp_path):
        client = self._client(monkeypatch, tmp_path)
        resp = client.post("/api/leverage", json={"leverage": 8})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["leverage"] == 3.0            # clamped, not the raw 8
        assert "3.0" in data["msg"]

    def test_below_min_leverage_echoes_clamped_value(self, monkeypatch, tmp_path):
        client = self._client(monkeypatch, tmp_path)
        resp = client.post("/api/leverage", json={"leverage": 0.2})
        assert resp.status_code == 200
        assert resp.get_json()["leverage"] == 1.0  # clamped up to floor

    def test_invalid_input_returns_400(self, monkeypatch, tmp_path):
        client = self._client(monkeypatch, tmp_path)
        resp = client.post("/api/leverage", json={"leverage": "abc"})
        assert resp.status_code == 400
        assert resp.get_json()["ok"] is False


class TestMomentumHoldReset:
    """#165: the momentum-hold SL-delay budget must reset once price recovers
    above the SL, so it grants N ticks of grace *per contiguous dip episode*,
    not once over the whole position lifetime."""

    def _strategy(self, **overrides):
        from strategies.grid import GridStrategy
        from strategies.grid_params import GridParams
        params = GridParams.from_dict(
            {"sl_mode": "floor", "leverage": 1.0,
             "momentum_hold_score": 0.35, "momentum_hold_max": 1, **overrides})
        return GridStrategy([{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}],
                            ml_enabled=False, params=params)

    def _setup(self, strategy):
        from core.context import MarketContext
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        state = strategy.get_state("SOL/USD")
        state._atr = 2.0
        # One open long grid position with a sell/SL leg.
        state.orders = {
            "c1": {"side": "sell", "price": 105.0, "qty": 1.0, "filled": False,
                   "bought_at": 100.0, "sl_price": 98.0, "momentum_holds": 0,
                   "leverage": 1.0},
        }
        state._direction_score = 0.9  # strongly bullish → holds are granted
        return ctx, state

    def test_recovery_resets_hold_budget(self):
        strategy = self._strategy()
        ctx, state = self._setup(strategy)

        # Dip #1: below SL → grace granted, no SL fire.
        strategy._check_position_stops("SOL/USD", 97.0, state, ctx)
        assert "c1" in state.orders, "first dip must be held, not stopped"
        assert state.orders["c1"]["momentum_holds"] == 1

        # Recovery above SL → budget must reset to 0.
        strategy._check_position_stops("SOL/USD", 101.0, state, ctx)
        assert state.orders["c1"]["momentum_holds"] == 0, "recovery must reset holds"

        # Dip #2 (an independent episode): must be held again, not stop out.
        strategy._check_position_stops("SOL/USD", 97.0, state, ctx)
        assert "c1" in state.orders, "independent later dip must still get grace"
        assert state.orders["c1"]["momentum_holds"] == 1

    def test_contiguous_dip_still_exhausts_budget(self):
        # Without any recovery, the budget is still finite: the 2nd contiguous
        # tick below SL stops out (max=1). Guards against the reset masking the cap.
        strategy = self._strategy()
        ctx, state = self._setup(strategy)
        strategy._check_position_stops("SOL/USD", 97.0, state, ctx)  # held (hold=1)
        assert "c1" in state.orders
        strategy._check_position_stops("SOL/USD", 97.0, state, ctx)  # budget spent → SL
        assert "c1" not in state.orders, "contiguous dip must still stop out after max"


class TestDirectionalDisabledInConfig:
    """#188: config/grid_params.json must keep the directional path disabled.
    directional trades bypass the broker (#33/#51) — their PnL is invisible to the
    equity curve and daily-drawdown brake — and the last OOS backtest showed a 12%
    win rate. The shipped config drifted to true; this locks it back to false."""

    def test_config_directional_disabled(self):
        import json
        from pathlib import Path
        cfg = json.loads((Path(__file__).parents[1] / "config" / "grid_params.json").read_text())
        assert cfg.get("directional_enabled") is False, \
            "config/grid_params.json must ship with directional_enabled=false (#188)"

    def test_loaded_params_have_directional_off(self):
        import json
        from pathlib import Path
        from strategies.grid_params import GridParams
        cfg = json.loads((Path(__file__).parents[1] / "config" / "grid_params.json").read_text())
        params = GridParams.from_dict(cfg)
        assert params.directional_enabled is False


class TestPaperBalanceRestorePartialOverlap:
    """#193: restoring persisted paper balances must not be all-or-nothing.

    A single coin toggle (#184) or any `symbols` edit changes the active set.
    The restore path in main.py used to require `set(saved) == set(symbols)`
    and would otherwise discard *every* balance, resetting all buckets to
    initial/len(active) and wiping accumulated equity + the drawdown baseline.
    The guard was dropped in favour of PaperBroker.load_balances' already
    selective semantics — this locks that contract in."""

    def _broker(self, symbols, balance=1000.0):
        from execution.paper import PaperBroker
        return PaperBroker(initial_balance=balance, symbols=symbols)

    def test_partial_overlap_preserves_unchanged_buckets(self):
        # Active set gained "C" and dropped "D" vs. the persisted session.
        broker = self._broker(["A/USD", "B/USD", "C/USD"], balance=900.0)
        fresh_default = 900.0 / 3  # 300 per active coin
        saved = {"A/USD": 500.0, "B/USD": 620.0, "D/USD": 999.0}
        broker.load_balances(saved)
        # Unchanged coins restore to their persisted value…
        assert broker._sym_balance("A/USD") == 500.0
        assert broker._sym_balance("B/USD") == 620.0
        # …the newly enabled coin keeps its fresh initial/len(active) default…
        assert broker._sym_balance("C/USD") == fresh_default
        # …and a now-removed coin's balance is PRESERVED as an orphan bucket, not
        # discarded: dropping it wiped real paper capital from equity and could
        # trip the deposit-anchored drawdown FREEZE on restart (#212 supersedes the
        # original "never leaked in" assertion — the orphan is real money, not a leak).
        assert broker._balances["D/USD"] == 999.0

    def test_restore_does_not_reset_to_startcapital(self):
        # The regression: after a toggle the total equity must reflect the
        # restored balances, not snap back to the fresh initial pool.
        broker = self._broker(["A/USD", "B/USD"], balance=1000.0)
        broker.load_balances({"A/USD": 700.0, "B/USD": 800.0})
        assert broker.get_balance() == 1500.0  # not 1000.0


class TestOrphanBalancePreservedOnRestart:
    """#212: disabling a coin (#184) or removing it from config.yaml:symbols must
    NOT wipe its accumulated paper capital from equity on the next restart.

    The persisted balance of the removed coin ("orphan") was silently dropped by
    load_balances' `if sym in self._balances` guard. With deposit-anchored drawdown
    (#132) the resulting phantom equity drop could instantly trip the daily-drawdown
    FREEZE and block new buys on every remaining coin — despite zero real loss.
    """

    def _broker(self, symbols, balance):
        from execution.paper import PaperBroker
        return PaperBroker(initial_balance=balance, symbols=symbols)

    def test_orphan_capital_stays_in_equity_after_disable(self):
        # 5 coins, 1000 deposit, all flat at 200. Disable 1 → restart with 4 active.
        deposit = 1000.0
        saved = {"A/USD": 200.0, "B/USD": 200.0, "C/USD": 200.0,
                 "D/USD": 200.0, "E/USD": 200.0}
        active = ["A/USD", "B/USD", "C/USD", "D/USD"]  # E disabled
        broker = self._broker(active, deposit)  # fresh buckets = 1000/4 = 250 each
        broker.load_balances(saved)
        # Equity must stay at the true 1000, NOT fall to 800 (the phantom-loss bug).
        assert broker.get_balance() == pytest.approx(1000.0)

    def test_no_false_drawdown_freeze(self):
        # The concrete freeze scenario from #212: baseline = deposit = 1000,
        # max_daily_drawdown = 10%. After disable+restart the equity must not dip
        # below the 900 freeze line purely from the orphan drop.
        deposit = 1000.0
        max_dd = 0.10
        saved = {f"{c}/USD": 200.0 for c in "ABCDE"}
        broker = self._broker(["A/USD", "B/USD", "C/USD", "D/USD"], deposit)
        broker.load_balances(saved)
        dd = (broker.get_balance() - deposit) / deposit
        assert dd > -max_dd, "orphan drop must not synthesise a >10% drawdown"

    def test_orphan_survives_second_restart_via_persistence(self):
        # Persistence saves dict(broker._balances) (engine.py:719/792). The orphan
        # must be in that dict so it is not lost on the *next* restart either.
        saved = {f"{c}/USD": 200.0 for c in "ABCDE"}
        broker = self._broker(["A/USD", "B/USD", "C/USD", "D/USD"], 1000.0)
        broker.load_balances(saved)
        persisted = dict(broker._balances)          # what engine writes to the DB
        assert persisted.get("E/USD") == 200.0

        # Second restart: same reduced active set, load the just-persisted dict.
        broker2 = self._broker(["A/USD", "B/USD", "C/USD", "D/USD"], 1000.0)
        broker2.load_balances(persisted)
        assert broker2.get_balance() == pytest.approx(1000.0)

    def test_reenable_restores_orphan_into_active_bucket(self):
        # Re-enabling E (back in the active set) must restore its saved value into
        # the tradeable bucket, not the fresh initial/len default.
        saved = {f"{c}/USD": 200.0 for c in "ABCD"}
        saved["E/USD"] = 260.0  # E had grown before being disabled
        active = ["A/USD", "B/USD", "C/USD", "D/USD", "E/USD"]  # E re-enabled
        broker = self._broker(active, 1000.0)  # fresh buckets = 200 each
        broker.load_balances(saved)
        assert broker._sym_balance("E/USD") == 260.0  # restored, not 200

    def test_orphan_bucket_is_untraded_and_isolated(self):
        # The orphan must NOT leak into the #149 fallback pool (_balance stays 0),
        # so an unseeded/mismatched symbol can't transact against hidden cash.
        saved = {"A/USD": 300.0, "GONE/USD": 500.0}
        broker = self._broker(["A/USD"], 300.0)
        broker.load_balances(saved)
        assert broker._balance == 0.0
        # An unseeded symbol still sees no cash (buys rejected), isolation intact.
        assert broker._sym_balance("NEVER/USD") == 0.0


class TestSLCancelsRestingBrokerOrder:
    """#192: a strategy-side floor-SL must cancel its resting broker sell order
    directly, not rely on the engine's _sync_orders cancel loop — that loop is
    skipped while the coin is frozen / emergency-stopped (#34) / disabled (#184),
    whereas on_tick_safety (which fires the SL) always runs. A left-open sell
    order would otherwise fill on a later price recovery over the sell level and
    credit margin+PnL a SECOND time on top of the SL credit."""

    def _setup(self, monkeypatch):
        from strategies.grid import GridStrategy, _GridState
        from strategies.grid_params import GridParams
        from execution.paper import PaperBroker
        monkeypatch.setenv("GRIDBOT_BACKTEST", "1")  # skip dashboard logging
        strat = GridStrategy(
            [{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}],
            ml_enabled=False, params=GridParams(leverage=1.0),
        )
        broker = PaperBroker(initial_balance=100.0, symbols=["SOL/USD"])
        strat._broker = broker
        # A resting sell (TP) order living on the broker, mirrored in strategy state.
        broker.place_limit("SOL/USD", "sell", 110.0, 1.0, client_id="c1")
        state = _GridState("SOL/USD", 100.0, 6, 0.05)
        state.orders["c1"] = {
            "side": "sell", "price": 110.0, "qty": 1.0, "filled": False,
            "bought_at": 100.0, "sl_price": 99.0, "leverage": 1.0,
            "momentum_holds": 0,
        }
        strat._states["SOL/USD"] = state
        return strat, broker, state

    def test_sl_cancels_broker_order(self, monkeypatch):
        from core.context import MarketContext
        strat, broker, state = self._setup(monkeypatch)
        # price 98 < sl_price 99 → floor-SL fires
        strat._check_position_stops("SOL/USD", 98.0, state, MarketContext())
        assert broker._orders["c1"].status == "cancelled", \
            "resting broker sell order must be cancelled by the SL"

    def test_no_double_credit_on_recovery(self, monkeypatch):
        from core.context import MarketContext
        strat, broker, state = self._setup(monkeypatch)
        strat._check_position_stops("SOL/USD", 98.0, state, MarketContext())
        bal_after_sl = broker.get_balance()
        # Price later recovers above the old sell level — the orphaned order must
        # NOT fill (it was cancelled), so no second margin+PnL credit occurs.
        broker.update_price("SOL/USD", 99.0)   # advance tick
        fills = broker.update_price("SOL/USD", 111.0)
        assert fills == [], "cancelled sell order must not fill on recovery"
        assert broker.get_balance() == bal_after_sl, "no second credit after SL"


class TestWaitFillsTerminatesWithPreseededSells:
    """#197: graceful wait_fills must self-terminate once all *real* positions
    (filled buys with an open TP-sell) are closed.  Pre-seeded placeholder sells
    carry a synthetic `bought_at` and are regenerated on every rebuild, so they
    must NOT count as open inventory — otherwise the bot never stops."""

    def _run_one_tick(self, monkeypatch, orders):
        import types
        from core.engine import Engine
        from execution.paper import PaperBroker
        from core.context import MarketContext

        state = types.SimpleNamespace(total_profit=0.0, investment=100.0,
                                      grid_lines=[], orders=orders)

        class FakeStrategy:
            _broker = None
            def get_state(self, s): return state
            def on_tick_safety(self, s, p, ctx): pass
            def on_tick(self, s, p, ctx): pass
            def on_candle(self, s, df, ctx): pass

        broker = PaperBroker(initial_balance=100.0, symbols=["SOL/USD"])
        eng = Engine(FakeStrategy(), broker, ["SOL/USD"],
                     ctx=MarketContext(), initial_capital=100.0)
        eng._waiting_for_fills = True

        for name in ("_check_dashboard_stop", "_check_daily_drawdown",
                     "_reconcile_fills", "_log_equity", "_sync_orders"):
            monkeypatch.setattr(eng, name, lambda *a, **k: None)
        monkeypatch.setattr(eng, "_update_dashboard", lambda s, p: None)
        monkeypatch.setattr(eng, "_update_prediction_outcomes", lambda f: None)

        import data_fetcher
        monkeypatch.setattr(data_fetcher, "fetch_ticker", lambda s: {"last": 100.0})
        monkeypatch.setattr(data_fetcher, "fetch_ohlcv", lambda *a, **k: None)
        import core.engine as ce
        monkeypatch.setattr(ce.time, "sleep", lambda *a, **k: None)

        eng._loop_count = 1
        eng._tick()
        return eng

    def test_only_preseeded_sells_terminates(self, monkeypatch):
        # Grid full of placeholder walls, no real inventory → bot must stop.
        orders = {
            "c1": {"side": "sell", "price": 110.0, "qty": 1.0, "filled": False,
                   "bought_at": 100.0, "pre_seeded": True},
            "c2": {"side": "sell", "price": 112.0, "qty": 1.0, "filled": False,
                   "bought_at": 100.0, "pre_seeded": True},
        }
        eng = self._run_one_tick(monkeypatch, orders)
        assert eng._shutdown.is_running() is False, \
            "wait_fills must terminate when only pre_seeded sells remain (#197)"

    def test_real_open_position_keeps_running(self, monkeypatch):
        # One genuine filled buy with an open TP-sell → must keep running.
        orders = {
            "c1": {"side": "sell", "price": 110.0, "qty": 1.0, "filled": False,
                   "bought_at": 100.0, "pre_seeded": True},
            "real": {"side": "sell", "price": 105.0, "qty": 1.0, "filled": False,
                     "bought_at": 100.0},  # no pre_seeded → real inventory
        }
        eng = self._run_one_tick(monkeypatch, orders)
        assert eng._shutdown.is_running() is True, \
            "wait_fills must NOT terminate while a real position is still open"


class TestGetConnPragmas:
    """#163: get_conn() must enable WAL + a non-zero busy_timeout.

    Dashboard and bot are two OS processes writing data/trades.db concurrently.
    Without journal_mode=WAL a writer blocks all readers, and with the SQLite
    default busy_timeout=0 a second writer fails immediately ("database is
    locked") instead of waiting — dropping bot writes (trades/equity) or
    returning dashboard 500s. This locks the pragmas into get_conn()."""

    def _conn(self, tmp_path, monkeypatch):
        import dashboard.db as ddb
        monkeypatch.setattr(ddb, "DB_PATH", tmp_path / "trades.db")
        return ddb.get_conn()

    def test_wal_enabled(self, tmp_path, monkeypatch):
        con = self._conn(tmp_path, monkeypatch)
        try:
            mode = con.execute("PRAGMA journal_mode").fetchone()[0]
            assert str(mode).lower() == "wal", f"expected WAL, got {mode!r}"
        finally:
            con.close()

    def test_busy_timeout_nonzero(self, tmp_path, monkeypatch):
        con = self._conn(tmp_path, monkeypatch)
        try:
            timeout = con.execute("PRAGMA busy_timeout").fetchone()[0]
            assert int(timeout) >= 5000, f"busy_timeout must be >=5000ms, got {timeout}"
        finally:
            con.close()


class TestCapitalChangeClearsPaperBalances:
    """#220: changing initial_capital in paper mode must clear the persisted
    paper_balances, otherwise the deposit-anchored drawdown baseline (= new
    capital) drifts from the restored old cash → instant, effectively permanent
    drawdown FREEZE (and grids sized for the new, larger capital get rejected
    with 'insufficient balance'). An UNCHANGED value must keep the balances so
    accumulated paper equity is not wiped by a no-op save."""

    def _db(self, tmp_path, monkeypatch):
        import dashboard.db as ddb
        monkeypatch.setattr(ddb, "DB_PATH", tmp_path / "trades.db")
        return ddb

    def test_capital_change_clears_persisted_balances(self, tmp_path, monkeypatch):
        ddb = self._db(tmp_path, monkeypatch)
        ddb.set_initial_capital(1000.0)
        ddb.save_paper_balances({"SOL/USD": 180.0, "ETH/USD": 175.0})
        assert ddb.load_paper_balances() is not None

        changed = ddb.set_initial_capital(2000.0)   # simulate "deposit" 1000 -> 2000
        assert changed is True
        # the stale buckets must be gone so the next start reseeds fresh == 2000
        assert ddb.load_paper_balances() is None
        assert ddb.get_initial_capital() == 2000.0

    def test_unchanged_capital_keeps_balances(self, tmp_path, monkeypatch):
        ddb = self._db(tmp_path, monkeypatch)
        ddb.set_initial_capital(1000.0)
        ddb.save_paper_balances({"SOL/USD": 180.0, "ETH/USD": 175.0})

        changed = ddb.set_initial_capital(1000.0)   # re-saving the same value
        assert changed is False
        # accumulated paper equity must survive a no-op save
        assert ddb.load_paper_balances() == {"SOL/USD": 180.0, "ETH/USD": 175.0}

    def test_endpoint_reports_reset_flag(self, tmp_path, monkeypatch):
        self._db(tmp_path, monkeypatch)
        import dashboard.app as dapp
        dapp.app.config["TESTING"] = True
        client = dapp.app.test_client()

        client.post("/api/capital", json={"initial_capital": 1000})
        # a genuine change → reset True (paper account starts fresh next start)
        resp = client.post("/api/capital", json={"initial_capital": 2500})
        data = resp.get_json()
        assert data["ok"] is True and data["reset"] is True
        # re-saving the same value → no reset
        resp2 = client.post("/api/capital", json={"initial_capital": 2500})
        assert resp2.get_json()["reset"] is False


class TestApiStartPidAware:
    """#162: /api/bot/start must gate on _is_running() (PID-file aware), not only
    on the in-process _bot_process handle. A bot started externally via
    ./start.sh --bot appears as _bot_process=None in the dashboard process; a pure
    _bot_process guard would spawn a SECOND main.py and overwrite .bot.pid with the
    dead PID of the child that loses the singleton lock — leaving the real bot
    unkillable while /api/status reports 'stopped'."""

    def _client(self, tmp_path, monkeypatch):
        import dashboard.app as dapp
        monkeypatch.setattr(dapp, "_ROOT", tmp_path)
        monkeypatch.setattr(dapp, "_bot_process", None)
        monkeypatch.setattr(dapp, "set_status", lambda **k: None)
        spawned = []

        class _FakePopen:
            def __init__(self, *a, **k):
                self.pid = 424242
                spawned.append((a, k))

            def poll(self):
                return None

        monkeypatch.setattr(dapp.subprocess, "Popen", _FakePopen)
        return dapp.app.test_client(), spawned

    def test_start_refused_when_external_bot_alive(self, tmp_path, monkeypatch):
        import os
        client, spawned = self._client(tmp_path, monkeypatch)
        # external bot: no in-process handle, but a live PID in .bot.pid
        (tmp_path / ".bot.pid").write_text(str(os.getpid()))
        resp = client.post("/api/bot/start", json={})
        data = resp.get_json()
        assert data["ok"] is False
        assert not spawned, "must NOT spawn a second bot when one is already running"
        # the real bot's PID must be left untouched (not clobbered by a dead child)
        assert (tmp_path / ".bot.pid").read_text().strip() == str(os.getpid())

    def test_start_spawns_when_nothing_running(self, tmp_path, monkeypatch):
        client, spawned = self._client(tmp_path, monkeypatch)
        # no .bot.pid and no in-process handle → free to start
        resp = client.post("/api/bot/start", json={})
        data = resp.get_json()
        assert data["ok"] is True
        assert len(spawned) == 1
        assert (tmp_path / ".bot.pid").read_text().strip() == "424242"


# ── #210: blocked coins must still fill resting TP-sells ─────────────────────

class TestBlockedCoinStillFillsTP:
    """#210: an emergency-stopped (#34) or dashboard-disabled (#184) coin must keep
    filling its resting grid TP-sell limit orders — only OPENING new risk is blocked.

    process_paper_fills() is the only paper path that fills resting limit sells; while
    it lived inside the block_new_risk-gated _sync_orders, a blocked coin could exit
    only via SL, contradicting the documented "SL/TP still active" and trapping an
    emergency-stopped coin in a loss-only one-way street. The fix pulls the fill out of
    the gate so it runs on block_new_risk — but NOT during the daily-drawdown freeze,
    whose only-SL one-way liquidation is intentional (#90, parked in #171).
    """

    class _Strat:
        """Minimal strategy: records fills, and FAILS loudly if the engine tries to
        open new risk (on_tick) or sync orders (desired_orders) for a blocked coin."""
        _broker = None

        def __init__(self):
            self.fills = []
            # on_tick (new-risk decisions) is skipped for blocked AND frozen coins.
            self.allow_on_tick = True
            # #222/#230: _sync_orders' CANCEL half runs for a BLOCKED coin AND during a
            # daily-drawdown FREEZE (only the PLACE half is skipped), so desired_orders
            # may be queried in both states.  It is skipped only in a hypothetical state
            # where the engine chooses not to sync at all (kept for explicitness).
            self.allow_sync = True

        def init(self, symbols, ctx):
            pass

        def get_state(self, sym):
            return None

        def on_tick_safety(self, sym, price, ctx):
            pass

        def on_tick(self, sym, price, ctx):
            if not self.allow_on_tick:
                raise AssertionError("on_tick must not run for a blocked/frozen coin")

        def desired_orders(self, sym, price, ctx):
            if not self.allow_sync:
                raise AssertionError("_sync_orders was not expected to run here")
            return []

        def on_fill(self, fill, ctx):
            self.fills.append(fill)

    def _engine(self, monkeypatch, price, disabled=False, frozen=False):
        from core.engine import Engine
        from core.context import MarketContext
        from execution.paper import PaperBroker
        import data_fetcher

        sym = "SOL/USD"
        broker = PaperBroker(initial_balance=1000.0, symbols=[sym])
        ctx = MarketContext()
        ctx.set_freeze(frozen)
        strat = self._Strat()
        strat.allow_on_tick = not (disabled or frozen)
        # #222/#230: the cancel half of _sync_orders runs for blocked coins AND while
        # frozen, so desired_orders is queried in every state.
        strat.allow_sync = True
        eng = Engine(strat, broker, [sym], ctx=ctx)

        # Neutralise all heavy I/O so a single _tick() runs in isolation.
        monkeypatch.setattr(data_fetcher, "fetch_ticker", lambda s: {"last": price})
        for m in ("_check_dashboard_stop", "_refresh_coin_settings", "_refresh_btc",
                  "_refresh_funding", "_refresh_correlations", "_check_daily_drawdown",
                  "_update_dashboard", "_log_equity", "_update_prediction_outcomes"):
            monkeypatch.setattr(eng, m, lambda *a, **k: None)

        if disabled:
            eng._disabled_coins = {sym}
        eng._loop_count = 1  # avoid the recheck(%5)/rebuild(%60) cycles
        return eng, broker, strat, sym

    def _place_resting_tp(self, broker, sym, qty=1.0, bought_at=100.0, sell_price=104.0):
        broker.place_limit(
            symbol=sym, side="sell", price=sell_price, qty=qty, post_only=True,
            client_id="tp1", meta={"bought_at": bought_at, "leverage": 1.0},
        )

    def test_disabled_coin_still_fills_resting_tp_sell(self, monkeypatch):
        # price rises above the TP → the resting sell must fill even though the coin
        # is disabled (block_new_risk). Balance is credited margin + P&L.
        eng, broker, strat, sym = self._engine(monkeypatch, price=105.0, disabled=True)
        self._place_resting_tp(broker, sym)
        bal_before = broker._sym_balance(sym)

        eng._tick()

        assert broker._orders["tp1"].status == "filled", "disabled coin TP-sell must fill"
        assert broker._sym_balance(sym) > bal_before, "TP fill must credit the balance"
        assert len(strat.fills) == 1 and strat.fills[0].side == "sell"

    def test_emergency_and_frozen_semantics_diverge(self, monkeypatch):
        # The daily-drawdown FREEZE keeps its intentional only-SL behaviour: the
        # resting TP-sell must NOT fill while frozen (guards against over-reaching the
        # #210 fix into the deliberately-parked #90/freeze path).  #230: the cancel
        # half of _sync_orders now runs while frozen (retracting stale rebuilt walls),
        # but process_paper_fills stays gated off, so no resting order fills here.
        eng, broker, strat, sym = self._engine(monkeypatch, price=105.0, frozen=True)
        self._place_resting_tp(broker, sym)
        bal_before = broker._sym_balance(sym)

        eng._tick()

        assert broker._orders["tp1"].status == "open", "frozen coin TP-sell must NOT fill"
        assert broker._sym_balance(sym) == bal_before
        assert strat.fills == []

    def test_normal_coin_fills_and_runs_sync(self, monkeypatch):
        # Sanity: a non-blocked coin fills its TP and DOES run on_tick/_sync_orders.
        eng, broker, strat, sym = self._engine(monkeypatch, price=105.0)
        self._place_resting_tp(broker, sym)
        bal_before = broker._sym_balance(sym)

        eng._tick()  # would raise from _Strat if new-risk paths were skipped

        assert broker._orders["tp1"].status == "filled"
        assert broker._sym_balance(sym) > bal_before

    def test_blocked_coin_resting_buy_does_NOT_fill(self, monkeypatch):
        # A blocked coin keeps its resting BUY orders (they aren't cancelled because
        # _sync_orders stays gated).  update_price() fills any order the price crosses,
        # so a dip must NOT be allowed to fill a resting buy — that would OPEN a fresh
        # long on a coin whose contract is "new buys halted": averaging down into an
        # emergency-stopped loser (#34) or buying a dashboard-disabled coin (#184).
        # Guards against #210 over-reaching from the sell (exit) path into the buy
        # (entry) path.  price DROPS below the resting buy → it must stay open.
        eng, broker, strat, sym = self._engine(monkeypatch, price=95.0, disabled=True)
        broker.place_limit(
            symbol=sym, side="buy", price=100.0, qty=1.0, post_only=True,
            client_id="buy1", meta={"leverage": 1.0},
        )
        bal_before = broker._sym_balance(sym)

        eng._tick()

        assert broker._orders["buy1"].status == "open", \
            "blocked coin resting BUY must NOT fill (would open new risk)"
        assert broker._sym_balance(sym) == bal_before, "no margin may be deducted"
        assert strat.fills == []

    def test_blocked_coin_fills_tp_but_not_buy_same_tick(self, monkeypatch):
        # Combined: with both a resting TP-sell (below price) and a resting buy (above
        # a rising price would not trigger it, so use a wide buy) the blocked coin fills
        # ONLY the sell.  price=105 → TP@104 fills, buy@110 (still >= price so it would
        # trigger as buy since 105<=110) must be skipped by sells_only.
        eng, broker, strat, sym = self._engine(monkeypatch, price=105.0, disabled=True)
        self._place_resting_tp(broker, sym)  # sell @ 104
        broker.place_limit(
            symbol=sym, side="buy", price=110.0, qty=1.0, post_only=True,
            client_id="buy1", meta={"leverage": 1.0},
        )

        eng._tick()

        assert broker._orders["tp1"].status == "filled", "TP-sell must still fill"
        assert broker._orders["buy1"].status == "open", "resting buy must be skipped"
        assert [f.side for f in strat.fills] == ["sell"]


class TestBlockedCoinCancelsStaleWalls:
    """#222: a blocked coin (emergency-stop #34 / dashboard-disabled #184) must still
    CANCEL stale broker orders that a grid rebuild has dropped — pre-seeded sell walls
    are regenerated with fresh client_ids every rebuild, so the old walls become
    orphaned relative to state.orders.  Previously _sync_orders was fully gated on the
    block path, so the CANCEL half never ran: the old walls lingered `open` on the
    broker and process_paper_fills filled them a tick later → on_fill orphan branch →
    a phantom profit-only credit inflated equity while state.total_profit stayed put,
    plus a never-terminal order leak.  The fix decouples the halves: CANCEL always
    runs, PLACE (new risk) stays gated.  Real filled-buy TP-sells keep their cid across
    rebuild, so they remain in `desired` and are NOT cancelled."""

    class _Strat:
        def __init__(self, desired):
            self._desired = desired

        def desired_orders(self, sym, price, ctx):
            return list(self._desired)

    def _engine(self, desired):
        from core.engine import Engine
        from core.context import MarketContext
        from execution.paper import PaperBroker
        sym = "SOL/USD"
        broker = PaperBroker(initial_balance=1000.0, symbols=[sym])
        eng = Engine(self._Strat(desired), broker, [sym], ctx=MarketContext())
        return eng, broker, sym

    def _fresh_wall(self, sym):
        from core.strategy import Order
        # a post-rebuild pre-seeded wall carrying a NEW client_id
        return Order(symbol=sym, side="sell", price=104.0, qty=1.0,
                     client_id="fresh", post_only=True, meta={})

    def test_blocked_coin_cancels_stale_wall_without_placing_new(self):
        sym = "SOL/USD"
        eng, broker, sym = self._engine([self._fresh_wall(sym)])
        # a STALE wall the rebuild dropped: tracked as active AND resting on the broker
        stale = broker.place_limit(symbol=sym, side="sell", price=103.0, qty=1.0,
                                   post_only=True, client_id="stale", meta={})
        eng._active_orders[sym] = {"stale": stale}

        eng._sync_orders(sym, 100.0, place_new=False)   # blocked coin path

        # the stale wall is retracted → it can no longer orphan-fill a tick later
        assert broker._orders["stale"].status == "cancelled"
        assert "stale" not in eng._active_orders[sym]
        # PLACE half is skipped → no new risk opened for the blocked coin
        assert "fresh" not in broker._orders
        assert "fresh" not in eng._active_orders[sym]

    def test_unblocked_coin_places_fresh_wall(self):
        # sanity: with place_new=True (normal coin) the fresh wall IS placed and the
        # stale one still cancelled — proves place_new is the only gated half.
        sym = "SOL/USD"
        eng, broker, sym = self._engine([self._fresh_wall(sym)])
        stale = broker.place_limit(symbol=sym, side="sell", price=103.0, qty=1.0,
                                   post_only=True, client_id="stale", meta={})
        eng._active_orders[sym] = {"stale": stale}

        eng._sync_orders(sym, 100.0, place_new=True)

        assert broker._orders["stale"].status == "cancelled"
        assert broker._orders["fresh"].status == "open"
        assert "fresh" in eng._active_orders[sym]

    def test_real_tp_sell_survives_block_path_cancel(self):
        # A real filled-buy TP-sell keeps its cid across rebuild, so it stays in
        # `desired` and must NOT be cancelled on the block path — only stale walls go.
        from core.strategy import Order
        sym = "SOL/USD"
        keep = Order(symbol=sym, side="sell", price=104.0, qty=1.0,
                     client_id="tp_real", post_only=True, meta={"bought_at": 100.0})
        eng, broker, sym = self._engine([keep])
        tp = broker.place_limit(symbol=sym, side="sell", price=104.0, qty=1.0,
                                post_only=True, client_id="tp_real", meta={})
        stale = broker.place_limit(symbol=sym, side="sell", price=103.0, qty=1.0,
                                   post_only=True, client_id="stale", meta={})
        eng._active_orders[sym] = {"tp_real": tp, "stale": stale}

        eng._sync_orders(sym, 100.0, place_new=False)

        assert broker._orders["tp_real"].status == "open", "real TP-sell must survive"
        assert "tp_real" in eng._active_orders[sym]
        assert broker._orders["stale"].status == "cancelled", "stale wall must go"


class TestFreezeCancelsStaleWalls:
    """#230: the SAME stale-wall orphan hazard as #222, but across the daily-drawdown
    FREEZE boundary.  During a freeze, process_paper_fills is gated off (correct), but
    setup_grid keeps rebuilding — regenerating pre-seeded walls / resting buys with
    fresh cids — while _sync_orders (the only canceller) used to be fully freeze-gated.
    So the OLD broker orders survived the multi-tick freeze un-cancelled and, on the
    freeze-LIFT tick, process_paper_fills (which runs BEFORE the cancel) filled them
    against a cid state.orders no longer knew → on_fill orphan branch: a stale sell
    books a phantom profit-only credit while total_profit stays put; a stale buy
    deducts margin for a position that is never tracked (phantom long / margin leak).

    The fix runs the CANCEL half of _sync_orders unconditionally (even while frozen)
    and gates only the PLACE half on freeze, so stale orders are retracted while the
    freeze is still active and cannot orphan-fill at lift.  Mirrors #222 for the
    block path; the freeze case was the remaining gap."""

    class _Strat:
        def __init__(self, desired):
            self._desired = desired

        def desired_orders(self, sym, price, ctx):
            return list(self._desired)

    def _engine(self, desired, frozen):
        from core.engine import Engine
        from core.context import MarketContext
        from execution.paper import PaperBroker
        sym = "SOL/USD"
        broker = PaperBroker(initial_balance=1000.0, symbols=[sym])
        ctx = MarketContext()
        ctx.set_freeze(frozen)
        eng = Engine(self._Strat(desired), broker, [sym], ctx=ctx)
        return eng, broker, sym

    def _fresh_wall(self, sym):
        from core.strategy import Order
        return Order(symbol=sym, side="sell", price=104.0, qty=1.0,
                     client_id="fresh", post_only=True, meta={})

    def test_frozen_coin_cancels_stale_wall_without_placing_new(self):
        # place_new must be False while frozen (the caller passes
        # place_new = not block_new_risk and not is_frozen()); the cancel half still
        # runs, so the stale wall is retracted but no fresh wall is placed.
        sym = "SOL/USD"
        eng, broker, sym = self._engine([self._fresh_wall(sym)], frozen=True)
        stale = broker.place_limit(symbol=sym, side="sell", price=103.0, qty=1.0,
                                   post_only=True, client_id="stale", meta={})
        eng._active_orders[sym] = {"stale": stale}

        eng._sync_orders(sym, 100.0, place_new=False)   # frozen coin path

        assert broker._orders["stale"].status == "cancelled", \
            "stale wall must be retracted during freeze so it cannot orphan-fill at lift"
        assert "stale" not in eng._active_orders[sym]
        assert "fresh" not in broker._orders, "no new risk may be placed while frozen"

    def test_full_tick_while_frozen_cancels_stale_wall(self, monkeypatch):
        # End-to-end via _tick(): a frozen coin's _sync_orders cancel half runs, so a
        # stale tracked wall is retracted and never orphan-fills.  The stale sell sits
        # ABOVE the price so process_paper_fills would not fill it this tick anyway; the
        # point is that it is GONE from the broker after the frozen tick.
        from core.engine import Engine
        from core.context import MarketContext
        from execution.paper import PaperBroker
        import data_fetcher

        sym = "SOL/USD"
        broker = PaperBroker(initial_balance=1000.0, symbols=[sym])
        ctx = MarketContext()
        ctx.set_freeze(True)
        # desired is empty (rebuild dropped the wall) → the tracked stale wall is undesired
        eng = Engine(self._Strat([]), broker, [sym], ctx=ctx)

        monkeypatch.setattr(data_fetcher, "fetch_ticker", lambda s: {"last": 100.0})
        for m in ("_check_dashboard_stop", "_refresh_coin_settings", "_refresh_btc",
                  "_refresh_funding", "_refresh_correlations", "_check_daily_drawdown",
                  "_update_dashboard", "_log_equity", "_update_prediction_outcomes"):
            monkeypatch.setattr(eng, m, lambda *a, **k: None)
        eng._loop_count = 1  # avoid recheck/rebuild cycles

        stale = broker.place_limit(symbol=sym, side="sell", price=104.0, qty=1.0,
                                   post_only=True, client_id="stale", meta={})
        eng._active_orders[sym] = {"stale": stale}

        eng._tick()

        assert broker._orders["stale"].status == "cancelled", \
            "stale wall must be cancelled on the frozen tick (#230)"
        assert "stale" not in eng._active_orders.get(sym, {})

    def test_real_tp_sell_survives_freeze_path_cancel(self):
        # A real filled-buy TP-sell keeps its cid across rebuild → stays in `desired`
        # and must NOT be cancelled on the freeze path, so the exit stays open.
        from core.strategy import Order
        sym = "SOL/USD"
        keep = Order(symbol=sym, side="sell", price=104.0, qty=1.0,
                     client_id="tp_real", post_only=True, meta={"bought_at": 100.0})
        eng, broker, sym = self._engine([keep], frozen=True)
        tp = broker.place_limit(symbol=sym, side="sell", price=104.0, qty=1.0,
                                post_only=True, client_id="tp_real", meta={})
        stale = broker.place_limit(symbol=sym, side="sell", price=103.0, qty=1.0,
                                   post_only=True, client_id="stale", meta={})
        eng._active_orders[sym] = {"tp_real": tp, "stale": stale}

        eng._sync_orders(sym, 100.0, place_new=False)

        assert broker._orders["tp_real"].status == "open", "real TP-sell must survive freeze"
        assert "tp_real" in eng._active_orders[sym]
        assert broker._orders["stale"].status == "cancelled", "stale wall must go"


class TestWaitFillsBlocksRestingBuyFill:
    """#214: the graceful "wait_fills" wind-down promises "no new buys", but it does
    NOT set block_new_risk (it is neither an emergency-stop nor a disable). On the
    activation tick process_paper_fills runs BEFORE _sync_orders cancels the now-
    undesired resting BUYs, so without gating sells_only on the wait_fills latch a
    resting BUY the price crosses would fill and OPEN a fresh long during the
    shutdown. Resting TP-sells (the exit path) must still fill.
    """

    class _Strat:
        """Minimal strategy: records fills. During wait_fills block_new_risk is
        False, so on_tick / desired_orders DO run — they must not raise."""
        _broker = None

        def __init__(self):
            self.fills = []

        def init(self, symbols, ctx):
            pass

        def get_state(self, sym):
            return None

        def on_tick_safety(self, sym, price, ctx):
            pass

        def on_tick(self, sym, price, ctx):
            pass

        def desired_orders(self, sym, price, ctx):
            # wait_fills → sell-only: no new orders desired (mirrors the sell_only latch)
            return []

        def on_fill(self, fill, ctx):
            self.fills.append(fill)

    def _engine(self, monkeypatch, price):
        from core.engine import Engine
        from core.context import MarketContext
        from execution.paper import PaperBroker
        import data_fetcher

        sym = "SOL/USD"
        broker = PaperBroker(initial_balance=1000.0, symbols=[sym])
        ctx = MarketContext()
        strat = self._Strat()
        eng = Engine(strat, broker, [sym], ctx=ctx)

        monkeypatch.setattr(data_fetcher, "fetch_ticker", lambda s: {"last": price})
        for m in ("_check_dashboard_stop", "_refresh_coin_settings", "_refresh_btc",
                  "_refresh_funding", "_refresh_correlations", "_check_daily_drawdown",
                  "_update_dashboard", "_log_equity", "_update_prediction_outcomes"):
            monkeypatch.setattr(eng, m, lambda *a, **k: None)

        eng._waiting_for_fills = True     # graceful wait_fills latch active
        eng._loop_count = 1               # avoid recheck(%5)/rebuild(%60) cycles
        return eng, broker, strat, sym

    def test_resting_buy_does_not_fill_during_wait_fills(self, monkeypatch):
        # price DROPS below a resting buy on the wait_fills activation tick → it must
        # NOT fill (would open a fresh long during the wind-down). Before the fix
        # process_paper_fills(sells_only=False) filled it.
        eng, broker, strat, sym = self._engine(monkeypatch, price=95.0)
        broker.place_limit(
            symbol=sym, side="buy", price=100.0, qty=1.0, post_only=True,
            client_id="buy1", meta={"leverage": 1.0},
        )
        bal_before = broker._sym_balance(sym)

        eng._tick()

        assert broker._orders["buy1"].status != "filled", \
            "resting BUY must NOT fill during wait_fills wind-down"
        assert not any(f.side == "buy" for f in strat.fills), "no buy fill callback"
        assert broker._sym_balance(sym) == bal_before, "no margin may be deducted"

    def test_resting_tp_sell_still_fills_during_wait_fills(self, monkeypatch):
        # The exit path must stay open: a resting TP-sell the price crosses still fills,
        # so existing positions can close and wait_fills can self-terminate.
        eng, broker, strat, sym = self._engine(monkeypatch, price=105.0)
        broker.place_limit(
            symbol=sym, side="sell", price=104.0, qty=1.0, post_only=True,
            client_id="tp1", meta={"bought_at": 100.0, "leverage": 1.0},
        )
        bal_before = broker._sym_balance(sym)

        eng._tick()

        assert broker._orders["tp1"].status == "filled", \
            "resting TP-sell must still fill during wait_fills"
        assert broker._sym_balance(sym) > bal_before, "TP fill must credit the balance"
        assert [f.side for f in strat.fills] == ["sell"]


class TestPreseededSellFeeNoPhantomBuyFee:
    """#215: filling a pre-seeded sell (an upper grid wall placed at setup WITHOUT
    a real buy) must charge the sell-side fee ONLY. Charging a round-trip fee books
    a phantom buy-fee (buy_price·qty·KRAKEN_FEE) into total_profit that the broker
    never mirrors (execution/paper.py:180-182 credits sell-fee only) — a one-way
    drift that under-reports realized P&L and biases both compounding and the
    emergency-stop. A real grid buy→sell round trip keeps its round-trip fee.
    """

    def _strategy(self):
        from strategies.grid import GridStrategy
        from strategies.grid_params import GridParams
        params = GridParams.from_dict({"sl_mode": "floor", "leverage": 1.0})
        return GridStrategy([{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}],
                            ml_enabled=False, params=params)

    def _setup(self, strategy, price=100.0, atr=2.0):
        from core.context import MarketContext
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        state = strategy.get_state("SOL/USD")
        state._atr = atr
        state.with_position = True  # mirror _refresh_prediction so buys seed too
        strategy.setup_grid("SOL/USD", price, ctx)
        return ctx, state

    def _fill(self, strategy, ctx, cid, order, side):
        from core.strategy import Fill
        strategy.on_fill(Fill(client_id=cid, symbol="SOL/USD", side=side,
                              price=order["price"], qty=order["qty"], fee=0.0,
                              ts=time.time()), ctx)

    def test_preseeded_fill_charges_sell_fee_only(self, monkeypatch):
        monkeypatch.setenv("GRIDBOT_BACKTEST", "1")  # skip dashboard/notifier
        from strategies.grid import KRAKEN_FEE
        strategy = self._strategy()
        ctx, state = self._setup(strategy)
        cid, order = next((c, o) for c, o in state.orders.items()
                          if o["side"] == "sell" and o.get("pre_seeded"))
        sell_price, bought_at, qty = order["price"], order["bought_at"], order["qty"]
        assert state.total_profit == 0.0

        self._fill(strategy, ctx, cid, order, "sell")

        expected = (sell_price - bought_at) * qty - sell_price * qty * KRAKEN_FEE
        assert state.total_profit == pytest.approx(expected)
        # The old bug charged the round-trip fee, i.e. an extra phantom buy-fee.
        roundtrip = ((sell_price - bought_at) * qty
                     - (sell_price + bought_at) * qty * KRAKEN_FEE)
        phantom_buy_fee = bought_at * qty * KRAKEN_FEE
        assert state.total_profit == pytest.approx(roundtrip + phantom_buy_fee)
        assert state.total_profit > roundtrip  # strictly better than the buggy value

    def test_preseeded_total_profit_matches_broker_credit(self, monkeypatch):
        # The whole point of #215: the strategy running-sum must move by the same
        # amount the PaperBroker credits for the same pre-seeded fill.
        monkeypatch.setenv("GRIDBOT_BACKTEST", "1")
        from strategies.grid import KRAKEN_FEE
        strategy = self._strategy()
        ctx, state = self._setup(strategy)
        cid, order = next((c, o) for c, o in state.orders.items()
                          if o["side"] == "sell" and o.get("pre_seeded"))
        sell_price, bought_at, qty = order["price"], order["bought_at"], order["qty"]
        self._fill(strategy, ctx, cid, order, "sell")
        # Broker's pre-seeded credit (execution/paper.py:180-182): sell-side fee only.
        broker_credit = (sell_price - bought_at) * qty - sell_price * qty * KRAKEN_FEE
        assert state.total_profit == pytest.approx(broker_credit)

    def test_real_position_keeps_roundtrip_fee(self, monkeypatch):
        # Guard the else-branch: a real grid buy→sell round trip must STILL be
        # charged the round-trip fee (its buy-fee left the broker balance at
        # buy-fill time; total_profit only moves on the sell, so it settles here).
        monkeypatch.setenv("GRIDBOT_BACKTEST", "1")
        from strategies.grid import KRAKEN_FEE
        strategy = self._strategy()
        ctx, state = self._setup(strategy)
        bcid, border = next((c, o) for c, o in state.orders.items()
                            if o["side"] == "buy")
        self._fill(strategy, ctx, bcid, border, "buy")
        state.total_profit = 0.0  # isolate the sell leg
        scid, sorder = next((c, o) for c, o in state.orders.items()
                            if o["side"] == "sell" and "sl_price" in o
                            and not o.get("pre_seeded"))
        sell_price, bought_at, qty = sorder["price"], sorder["bought_at"], sorder["qty"]
        self._fill(strategy, ctx, scid, sorder, "sell")
        expected = ((sell_price - bought_at) * qty
                    - (sell_price + bought_at) * qty * KRAKEN_FEE)
        assert state.total_profit == pytest.approx(expected)


class TestSellFillPreseededDriftAfterRebuild:
    """#235: a pre-seeded upper wall that survives a 15-min grid rebuild reuses
    its client_id while its bought_at/qty are re-written in state.orders to the
    CURRENT price/leverage — but the PaperBroker keeps (and credits paper cash
    from) the ORIGINAL values, because _sync_orders never re-places an already-
    active cid. _handle_sell_fill must book state.total_profit from what the
    broker ACTUALLY transacted (fill.qty / fill.meta), not the rewritten state
    entry, or total_profit and paper cash diverge — feeding the deposit-anchored
    drawdown brake (reads cash+MTM) and the emergency-stop / compounding (read
    total_profit) inconsistent realities. This is the sell-side counterpart of
    the #206/#208 buy-side fix.
    """

    def _setup(self, monkeypatch):
        from strategies.grid import GridStrategy, _GridState
        from strategies.grid_params import GridParams
        from execution.paper import PaperBroker
        from core.context import MarketContext

        monkeypatch.setenv("GRIDBOT_BACKTEST", "1")  # skip dashboard/notifier
        strat = GridStrategy(
            [{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}],
            ml_enabled=False,
            params=GridParams(sl_mode="floor", leverage=1.0,
                              trend_filter_enabled=False),
        )
        broker = PaperBroker(initial_balance=100.0, symbols=["SOL/USD"])
        strat._broker = broker
        ctx = MarketContext()
        monkeypatch.setattr(strat, "_lev", lambda: 1.0)

        state = _GridState("SOL/USD", 100.0, 6, 0.05)
        state.grid_lines = [95.0, 100.0, 105.0]
        strat._states["SOL/USD"] = state
        return strat, broker, ctx, state

    def test_total_profit_matches_broker_cash_after_wall_rebuild(self, monkeypatch):
        from strategies.grid import KRAKEN_FEE
        strat, broker, ctx, state = self._setup(monkeypatch)

        gp, cid = 105.0, "wall1"
        # ORIGINAL pre-seeded wall: first seeded when price=100 → bought_at=100.
        # This is the value the broker holds and credits paper cash from.
        orig_bought_at, orig_qty = 100.0, 0.10
        state.orders[cid] = {"side": "sell", "price": gp, "qty": orig_qty,
                             "filled": False, "bought_at": orig_bought_at,
                             "pre_seeded": True}
        state.price_to_id[gp] = cid
        # Place it in the broker exactly as desired_orders would (order dict → meta),
        # so the broker credit uses these ORIGINAL values.
        broker.place_limit(symbol="SOL/USD", side="sell", price=gp, qty=orig_qty,
                           client_id=cid,
                           meta={"bought_at": orig_bought_at, "pre_seeded": True,
                                 "leverage": 1.0})

        # A later rebuild recurs the same grid line, reuses the cid, and re-writes
        # bought_at to the CURRENT (drifted) price and qty to the current sizing.
        # The already-placed broker order is NOT touched (still 100.0 / 0.10).
        state.orders[cid]["bought_at"] = 104.0
        state.orders[cid]["qty"] = 0.30

        cash_before = broker._balances["SOL/USD"]
        fills = broker.update_price("SOL/USD", gp)   # wall triggers at 105
        assert len(fills) == 1 and fills[0].client_id == cid
        cash_delta = broker._balances["SOL/USD"] - cash_before
        fill_price = fills[0].price   # broker applies 3bps sell slippage

        assert state.total_profit == 0.0
        strat.on_fill(fills[0], ctx)

        # total_profit must move by exactly what the broker credited to cash.
        assert state.total_profit == pytest.approx(cash_delta)
        # …which equals the broker-consistent value from the ORIGINAL qty/bought_at.
        expected = (fill_price - orig_bought_at) * orig_qty - fill_price * orig_qty * KRAKEN_FEE
        assert state.total_profit == pytest.approx(expected)
        # The OLD bug booked from the rewritten state (bought_at 104, qty 0.30):
        # a materially different number. Guard against regressing to it.
        buggy = (fill_price - 104.0) * 0.30 - fill_price * 0.30 * KRAKEN_FEE
        assert abs(state.total_profit - buggy) > 1e-3

    def test_no_rebuild_case_is_unchanged(self, monkeypatch):
        # When state.orders was NOT rewritten (the common case), fill.qty/fill.meta
        # equal the state values, so booking is bit-identical to the old code path.
        from strategies.grid import KRAKEN_FEE
        strat, broker, ctx, state = self._setup(monkeypatch)

        gp, cid = 105.0, "wall2"
        bought_at, qty = 100.0, 0.10
        state.orders[cid] = {"side": "sell", "price": gp, "qty": qty,
                             "filled": False, "bought_at": bought_at,
                             "pre_seeded": True}
        state.price_to_id[gp] = cid
        broker.place_limit(symbol="SOL/USD", side="sell", price=gp, qty=qty,
                           client_id=cid,
                           meta={"bought_at": bought_at, "pre_seeded": True,
                                 "leverage": 1.0})

        fills = broker.update_price("SOL/USD", gp)
        fill_price = fills[0].price   # broker applies 3bps sell slippage
        strat.on_fill(fills[0], ctx)
        expected = (fill_price - bought_at) * qty - fill_price * qty * KRAKEN_FEE
        assert state.total_profit == pytest.approx(expected)


# ── Engine rebuild-orphan fill (#218) ────────────────────────────────────────

class TestRebuildOrphanFill:
    """Regression for #218: a resting broker order that the price crosses on a
    grid-rebuild tick must be FILLED AND BOOKED before setup_grid rebuilds
    state.orders and orphans its client_id.

    Under the old ordering (setup_grid → process_paper_fills), an out_of_range
    rebuild firing on the tick the price crosses a resting buy let the paper
    broker move cash (deduct margin) while GridStrategy.on_fill could no longer
    find the cid in state.orders and returned silently — margin deducted with no
    tracked position and no return-sell (phantom long / margin leak).

    The invariant checked here: after a rebuild tick, EVERY buy the broker filled
    is represented by exactly one tracked long position (a non-pre-seeded sell
    carrying bought_at) in state.orders. A leaked fill breaks this equality.
    """

    def _dummy_df(self, n=60):
        close = pd.Series(np.linspace(100, 100, n))
        return pd.DataFrame({
            "open": close, "high": close * 1.001,
            "low": close * 0.999, "close": close,
            "volume": np.full(n, 1000.0),
        })

    def _build(self, monkeypatch, tick_price):
        import data_fetcher
        import core.engine as eng_mod
        from core.engine import Engine
        from strategies.grid import GridStrategy
        from execution.paper import PaperBroker

        strat = GridStrategy(
            [{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}],
            ml_enabled=False,
        )
        broker = PaperBroker(initial_balance=1000.0, symbols=["SOL/USD"])
        eng = Engine(strat, broker, ["SOL/USD"])

        # No network, no dashboard/DB side effects, no real sleeps.
        monkeypatch.setattr(data_fetcher, "fetch_ticker",
                            lambda s: {"last": tick_price})
        monkeypatch.setattr(data_fetcher, "fetch_ohlcv",
                            lambda s, tf, n: self._dummy_df())
        monkeypatch.setattr(eng_mod.time, "sleep", lambda *a, **k: None)
        for m in ("_check_dashboard_stop", "_refresh_coin_settings",
                  "_check_daily_drawdown", "_update_dashboard", "_log_equity",
                  "_update_prediction_outcomes"):
            monkeypatch.setattr(eng, m, lambda *a, **k: None)

        # Build the grid at 100 and place the resting orders in the broker,
        # exactly as run()'s first pass + _sync_orders would.
        strat.init(["SOL/USD"], eng.ctx)
        state = strat.get_state("SOL/USD")
        state.with_position = True  # allow buy seeding (mirrors _refresh_prediction)
        strat.setup_grid("SOL/USD", 100.0, eng.ctx)
        eng._sync_orders("SOL/USD", 100.0)
        return eng, strat, broker, state

    def test_rebuild_tick_books_orphan_fills_no_margin_leak(self, monkeypatch):
        sym = "SOL/USD"
        # A dip below the grid bottom → out_of_range rebuild fires this tick and
        # the price crosses the resting buys.
        eng, strat, broker, state = self._build(monkeypatch, tick_price=90.0)

        resting_buys = [o for o in broker._orders.values()
                        if o.symbol == sym and o.side == "buy" and o.status == "open"]
        assert resting_buys, "precondition: broker holds resting buy orders"
        low = min(o.price for o in resting_buys)
        assert 90.0 < low, "tick price must cross at least one resting buy"

        # loop_count=41 → out_of_range rebuild is allowed (41 - 0 >= 40) while
        # avoiding the scheduled-rebuild / BTC / funding / on_candle cadences.
        eng._loop_count = 41
        eng._tick()

        filled_buys = [o for o in broker._orders.values()
                       if o.symbol == sym and o.side == "buy" and o.status == "filled"]
        positions = [o for o in state.orders.values()
                     if o.get("side") == "sell" and "bought_at" in o
                     and not o.get("pre_seeded")]

        assert filled_buys, "the dip must have filled at least one resting buy"
        # The core invariant: no filled buy was silently dropped by on_fill.
        assert len(positions) == len(filled_buys), (
            f"margin leak: {len(filled_buys)} buys filled but only "
            f"{len(positions)} positions tracked (orphaned fills)"
        )

    def test_process_paper_fills_runs_before_setup_grid(self, monkeypatch):
        # Directly pin the ordering the fix depends on: within a rebuild tick,
        # process_paper_fills must be invoked before setup_grid.
        eng, strat, broker, state = self._build(monkeypatch, tick_price=90.0)
        calls = []
        real_fills = eng.process_paper_fills
        real_setup = strat.setup_grid

        def spy_fills(*a, **k):
            calls.append("fills")
            return real_fills(*a, **k)

        def spy_setup(*a, **k):
            calls.append("setup")
            return real_setup(*a, **k)

        monkeypatch.setattr(eng, "process_paper_fills", spy_fills)
        monkeypatch.setattr(strat, "setup_grid", spy_setup)

        eng._loop_count = 41
        eng._tick()

        assert "fills" in calls and "setup" in calls
        assert calls.index("fills") < calls.index("setup"), (
            "process_paper_fills must run before setup_grid so a rebuild cannot "
            "orphan a fillable resting order (#218)"
        )


# ── Dashboard /api/capital running-guard (#224) ────────────────────────────────

class TestCapitalRunningGuard:
    """A capital change while the bot RUNS must be rejected, because the running
    engine re-persists its unchanged paper_balances every tick and on shutdown,
    which would clobber the paper_balances=NULL that #220 relies on (#224)."""

    def _client(self):
        from dashboard import app as dash_app
        dash_app.app.config["TESTING"] = True
        return dash_app, dash_app.app.test_client()

    def test_capital_change_rejected_while_running(self, monkeypatch):
        dash_app, client = self._client()
        monkeypatch.setattr(dash_app, "_is_running", lambda: True)

        called = {"set": False}

        def _spy_set(value):
            called["set"] = True
            return True

        # Endpoint imports set_initial_capital from dashboard.db lazily.
        import dashboard.db as db
        monkeypatch.setattr(db, "set_initial_capital", _spy_set)

        resp = client.post("/api/capital", json={"initial_capital": 2000})
        assert resp.status_code == 409
        body = resp.get_json()
        assert body["ok"] is False
        assert not called["set"], (
            "set_initial_capital() must NOT run while the bot is live — otherwise "
            "the engine re-persists over the paper_balances=NULL reset (#224)"
        )

    def test_capital_change_allowed_while_stopped(self, monkeypatch):
        dash_app, client = self._client()
        monkeypatch.setattr(dash_app, "_is_running", lambda: False)

        called = {"value": None}

        def _spy_set(value):
            called["value"] = value
            return True

        import dashboard.db as db
        monkeypatch.setattr(db, "set_initial_capital", _spy_set)

        resp = client.post("/api/capital", json={"initial_capital": 2000})
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["ok"] is True
        assert called["value"] == 2000.0


# ── Trailing-stop ratchets on the freeze/block safety path (#227) ───────────────

class TestTrailingStopOnSafetyPath:
    """During a daily-drawdown freeze / per-coin emergency-stop (#34) /
    dashboard-disable (#184) the engine skips on_tick but always calls
    on_tick_safety. The break-even/trail ratchet must still run there, or an open
    winner's SL never climbs and it gives back its whole book profit to the floor
    exactly when it matters most (#227)."""

    def _strategy(self, **overrides):
        from strategies.grid import GridStrategy
        from strategies.grid_params import GridParams
        params = GridParams.from_dict({"sl_mode": "floor", "leverage": 1.0, **overrides})
        return GridStrategy([{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}],
                            ml_enabled=False, params=params)

    def test_on_tick_safety_ratchets_trailing_stop(self):
        from core.context import MarketContext
        strategy = self._strategy()
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        state = strategy.get_state("SOL/USD")
        state._atr = 2.0  # ATR = $2

        cid = str(uuid.uuid4())
        state.orders[cid] = {
            "side": "sell", "price": 110.0, "qty": 1.0,
            "filled": False, "bought_at": 100.0,
            "sl_price": 96.0, "trailing_activated": False,
        }

        # Price is +1×ATR above entry (buy=100, ATR=2 → 102). on_tick_safety is the
        # ONLY hook the engine runs during a freeze/block. Before the #227 fix it did
        # not touch the trailing stop, so the SL stayed at 96.0.
        strategy.on_tick_safety("SOL/USD", 102.5, ctx)
        assert state.orders[cid]["trailing_activated"] is True, (
            "trailing stop must activate on the safety path during freeze/block (#227)"
        )
        assert state.orders[cid]["sl_price"] >= 100.0, (
            "SL must ratchet to break-even on on_tick_safety, not stay at the floor (#227)"
        )

    def test_safety_then_on_tick_ratchets_trailing_once(self):
        """The paired on_tick this same tick must not double-run the ratchet (#146)."""
        from core.context import MarketContext
        strategy = self._strategy()
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)

        calls = {"trail": 0}
        strategy._update_trailing_stops = lambda *a, **k: calls.__setitem__(
            "trail", calls["trail"] + 1)

        strategy.on_tick_safety("SOL/USD", 100.0, ctx)
        strategy.on_tick("SOL/USD", 100.0, ctx)
        assert calls["trail"] == 1, "trailing ratchet must run exactly once per tick (#146/#227)"

    def test_standalone_on_tick_still_ratchets_trailing(self):
        """on_tick called without a preceding safety tick still runs the ratchet."""
        from core.context import MarketContext
        strategy = self._strategy()
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)

        calls = {"trail": 0}
        strategy._update_trailing_stops = lambda *a, **k: calls.__setitem__(
            "trail", calls["trail"] + 1)

        strategy.on_tick("SOL/USD", 100.0, ctx)
        assert calls["trail"] == 1


# ── Smart-replenish never places a buy above market (#226) ──────────────────────

class TestReplenishNotAboveMarket:
    """The bullish smart-replenish branch placed the follow-up buy one grid line
    ABOVE the just-filled sell (= above market). A buy limit above market is
    immediately marketable and the PaperBroker fills it at the worse limit price,
    opening the new long already in the red. The replenish must be clamped to the
    sell fill price so it is never above market (#226)."""

    def _strategy(self, **overrides):
        from strategies.grid import GridStrategy
        from strategies.grid_params import GridParams
        params = GridParams.from_dict({"sl_mode": "floor", "leverage": 1.0, **overrides})
        return GridStrategy([{"symbol": "SOL/USD", "investment": 100.0, "levels": 5}],
                            ml_enabled=False, params=params)

    def _fill_sell(self, direction_score):
        from core.context import MarketContext
        from core.strategy import Fill
        strategy = self._strategy()
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        state = strategy.get_state("SOL/USD")
        state.grid_lines = [98.0, 100.0, 102.0, 104.0, 106.0]
        state.usdt_per_grid = 20.0
        state._direction_score = direction_score
        # Ensure the replenish branch actually runs regardless of trend-filter state.
        strategy._buys_allowed = lambda s: True

        sell_price = 102.0  # grid line idx 2; next higher line is 104.0 (above market)
        cid = str(uuid.uuid4())
        state.orders[cid] = {
            "side": "sell", "price": sell_price, "qty": 1.0,
            "filled": False, "bought_at": 100.0,
        }
        strategy.on_fill(Fill(client_id=cid, symbol="SOL/USD", side="sell",
                              price=sell_price, qty=1.0, fee=0.0, ts=time.time()), ctx)
        buys = [o for o in state.orders.values() if o["side"] == "buy"]
        assert len(buys) == 1, "exactly one replenish buy expected"
        return buys[0]["price"], sell_price

    def test_bullish_replenish_not_above_market(self):
        buy_price, sell_price = self._fill_sell(direction_score=0.5)
        assert buy_price <= sell_price, (
            f"replenish buy @ {buy_price} must not be placed above the market "
            f"(sell fill @ {sell_price}) — a buy limit above market fills at the "
            f"worse limit price (#226)"
        )

    def test_neutral_replenish_at_entry(self):
        # Non-bullish path is unchanged: replenish at the original entry (below market).
        buy_price, sell_price = self._fill_sell(direction_score=0.0)
        assert buy_price == 100.0


class TestHoldingSecondsEntryTs:
    """#105: every grid sell order must carry an ``entry_ts`` so
    ``_handle_sell_fill`` can measure a real holding duration.

    The original fix (commit 2b6f08c, 2026-07-02) regressed out of the tree in a
    lost merge batch, and the issue was wrongly closed as completed. Without the
    ``entry_ts`` key, ``order.get("entry_ts", time.time())`` fell back to *now*
    on every fill -> ``holding_seconds`` collapsed to ~0 for every ``grid_fill``
    trade, silently corrupting the hold-time analytics the optimizer and nightly
    tuner consume (``scripts/optimize.py`` wins_ht/loss_ht averaging).
    """

    def _strategy(self):
        from strategies.grid import GridStrategy
        return GridStrategy([{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}])

    def test_grid_build_stamps_entry_ts_on_preseeded_sells(self):
        from core.context import MarketContext
        strategy = self._strategy()
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        strategy.setup_grid("SOL/USD", 100.0, ctx)
        state = strategy.get_state("SOL/USD")

        preseeded = [o for o in state.orders.values()
                     if o["side"] == "sell" and o.get("pre_seeded")]
        assert preseeded, "grid build must create pre-seeded sell walls"
        assert all(o.get("entry_ts", 0) > 0 for o in preseeded), \
            "every pre-seeded sell must carry a positive entry_ts (#105)"

    def test_buy_fill_stamps_entry_ts_on_sell(self):
        from core.context import MarketContext
        from core.strategy import Fill
        strategy = self._strategy()
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        state = strategy.get_state("SOL/USD")
        state.with_position = True  # buys only seed when with_position (mirrors _refresh_prediction)
        strategy.setup_grid("SOL/USD", 100.0, ctx)

        buy_orders = [(cid, o) for cid, o in state.orders.items() if o["side"] == "buy"]
        assert buy_orders, "grid build must create resting buy orders"
        cid, order = buy_orders[0]
        sells_before = {c for c, o in state.orders.items() if o["side"] == "sell"}

        strategy.on_fill(
            Fill(client_id=cid, symbol="SOL/USD", side="buy",
                 price=order["price"], qty=order["qty"], fee=0.0, ts=time.time()),
            ctx,
        )

        new_sells = [o for c, o in state.orders.items()
                     if o["side"] == "sell" and c not in sells_before]
        assert new_sells, "a buy fill must create a matching TP sell order"
        assert all(o.get("entry_ts", 0) > 0 for o in new_sells), \
            "the sell created on a buy fill must carry a positive entry_ts (#105)"


class TestPaperBrokerOrderGC:
    """#135: PaperBroker used to keep every order it ever placed in ``_orders``
    forever and re-scan the whole map on every ``update_price`` tick -> unbounded
    memory growth and an O(all-orders-ever) per-tick scan in long paper/sweep
    runs. Terminal orders are now retired from the scanned open set and the
    retained-terminal map is bounded by a FIFO window.
    """

    def _broker(self):
        from execution.paper import PaperBroker
        return PaperBroker(initial_balance=1000.0, symbols=["SOL/USD"])

    def test_open_set_shrinks_on_fill_and_cancel(self):
        b = self._broker()
        b.place_limit("SOL/USD", "sell", 100.0, 1.0, client_id="s1",
                      meta={"bought_at": 90.0, "leverage": 1.0})
        b.place_limit("SOL/USD", "buy", 50.0, 1.0, client_id="b1",
                      meta={"leverage": 1.0})
        assert set(b._open_orders) == {"s1", "b1"}

        # Sell fills (price >= 100), the buy @50 does not (price 105 > 50).
        fills = b.update_price("SOL/USD", 105.0)
        assert [f.client_id for f in fills] == ["s1"]
        assert "s1" not in b._open_orders
        assert b._orders["s1"].status == "filled"  # still queryable
        assert b.get_open_orders("SOL/USD") == [b._orders["b1"]]

        # Cancel the resting buy → leaves the open set, stays queryable.
        assert b.cancel("b1") is True
        assert b._open_orders == {}
        assert b._orders["b1"].status == "cancelled"
        # cancel() is an idempotent no-op on an already-terminal id.
        assert b.cancel("b1") is False

    def test_terminal_map_is_bounded(self):
        from collections import deque
        b = self._broker()
        b._terminal_ids = deque(maxlen=10)  # shrink for a fast, deterministic test
        for i in range(50):
            cid = f"c{i}"
            b.place_limit("SOL/USD", "buy", 1.0, 1.0, client_id=cid, meta={"leverage": 1.0})
            b.cancel(cid)
        assert len(b._open_orders) == 0
        assert len(b._orders) <= 10, "retained terminal orders must be bounded (#135)"
        # The most recent terminal order is still queryable within the window.
        assert "c49" in b._orders and b._orders["c49"].status == "cancelled"
        # cancel() on an evicted id is a harmless no-op (guards the #192 SL path).
        assert b.cancel("c0") is False

    def test_live_order_still_fills_after_many_terminals(self):
        from collections import deque
        b = self._broker()
        b._terminal_ids = deque(maxlen=5)
        for i in range(20):
            cid = f"t{i}"
            b.place_limit("SOL/USD", "buy", 1.0, 1.0, client_id=cid, meta={"leverage": 1.0})
            b.cancel(cid)
        b.place_limit("SOL/USD", "sell", 100.0, 1.0, client_id="live",
                      meta={"bought_at": 90.0, "leverage": 1.0})
        fills = b.update_price("SOL/USD", 101.0)
        assert [f.client_id for f in fills] == ["live"]


class TestPaperBrokerReusedCidGC:
    """#237: the #135 terminal-order GC evicts the oldest terminal cid from
    ``_orders`` when the FIFO overflows, on the (false-for-this-codebase)
    assumption that a terminal cid is never read back. But client_ids ARE reused:
    ``setup_grid`` recycles a level's cid across rebuilds and ``_sync_orders``
    re-places it, so a cid can be OPEN again while an OLD terminal entry for the
    same cid still sits in the FIFO. Evicting it dropped the live order from
    ``_orders`` → ``cancel()`` silently no-oped on a still-fillable order → orphan
    fill (engine cancel-on-absent) or the #192 SL double-credit (a retracted sell
    fills a second time). A reused-and-live cid must stay cancellable regardless
    of GC pressure.
    """

    def _broker(self):
        from execution.paper import PaperBroker
        return PaperBroker(initial_balance=1000.0, symbols=["SOL/USD"])

    def test_reused_cid_stays_cancellable_after_gc_pressure(self):
        """End-to-end reproduction: place X, retire it, re-place X as a live
        resting order, then churn a full FIFO window of terminal orders. cancel(X)
        must still retract the live order (returned False on the old GC)."""
        from collections import deque
        b = self._broker()
        b._terminal_ids = deque(maxlen=5)

        # 1) Level gets cid X, then becomes undesired → cancelled → X goes terminal.
        b.place_limit("SOL/USD", "buy", 90.0, 1.0, client_id="X", meta={"leverage": 1.0})
        assert b.cancel("X") is True

        # 2) A later rebuild re-creates the level and reuses X as a live resting sell.
        b.place_limit("SOL/USD", "sell", 100.0, 1.0, client_id="X",
                      meta={"bought_at": 90.0, "leverage": 1.0})

        # 3) A full FIFO window of unrelated terminal orders churns through.
        for i in range(12):
            cid = f"c{i}"
            b.place_limit("SOL/USD", "buy", 1.0, 1.0, client_id=cid, meta={"leverage": 1.0})
            b.cancel(cid)

        # X is still live and must be cancellable — the whole point of the SL/orphan
        # cancel paths (#192 / #218 / #222 / #230) rely on this working.
        assert "X" in b._open_orders
        assert b.cancel("X") is True
        assert b._orders["X"].status == "cancelled"
        assert "X" not in b._open_orders
        # And once retracted it must NOT fill on a later price cross.
        assert b.update_price("SOL/USD", 101.0) == []

    def test_retire_never_evicts_a_live_reused_cid(self):
        """Directly exercise the _retire guard: force the pathological state where a
        cid sits in the terminal FIFO while ALSO being open, then trip an eviction.
        The live order must survive in ``_orders``."""
        from collections import deque
        b = self._broker()
        b._terminal_ids = deque(maxlen=3)

        # Live resting sell under cid X.
        b.place_limit("SOL/USD", "sell", 100.0, 1.0, client_id="X",
                      meta={"bought_at": 90.0, "leverage": 1.0})
        # Simulate a stale terminal entry for the SAME cid still sitting in the FIFO
        # (as it would before the place_limit cleanup, when a rebuild recycled it).
        b._terminal_ids.appendleft("X")
        b._terminal_set.add("X")

        # Fill the FIFO and trip an overflow so X reaches the eviction slot.
        for i in range(3):
            cid = f"t{i}"
            b.place_limit("SOL/USD", "buy", 1.0, 1.0, client_id=cid, meta={"leverage": 1.0})
            b.cancel(cid)

        # The live X order must NOT have been dropped from _orders.
        assert "X" in b._open_orders
        assert "X" in b._orders and b._orders["X"].status == "open"
        assert b.cancel("X") is True

    def test_cancel_falls_back_to_open_orders(self):
        """Even if a queryability eviction has already removed a live cid from
        ``_orders``, cancel() must still retract it via ``_open_orders`` so it
        cannot fill again and double-credit (#192)."""
        b = self._broker()
        b.place_limit("SOL/USD", "sell", 100.0, 1.0, client_id="X",
                      meta={"bought_at": 90.0, "leverage": 1.0})
        # Simulate the #135 eviction having dropped the live order from _orders.
        b._orders.pop("X", None)
        assert "X" in b._open_orders  # still live and fillable

        assert b.cancel("X") is True
        assert b._open_orders == {}
        # Retracted → a later cross must not produce a (double-crediting) fill.
        assert b.update_price("SOL/USD", 101.0) == []


class TestRemovePositionSingleLot:
    """#159: closing ONE grid lot (sell fill or stop-loss) must remove exactly
    that lot from the risk context, not the whole DCA cohort.

    ``remove_position(symbol, "grid")`` used to drop *every* grid position for
    the symbol, so after the first sell/SL RiskManager saw 0 open lots while
    several DCA buys were still open — ``open_position_count`` and
    ``symbol_position_usdt`` under-counted and the exposure/correlation caps
    were silently defeated. The fix keys each lot by its TP sell order's cid and
    removes only the matching one.
    """

    def _strategy(self, **overrides):
        from strategies.grid import GridStrategy
        from strategies.grid_params import GridParams
        params = GridParams.from_dict({"sl_mode": "floor", "leverage": 1.0, **overrides})
        return GridStrategy([{"symbol": "SOL/USD", "investment": 100.0, "levels": 5}],
                            ml_enabled=False, params=params)

    def _fill_two_buys(self):
        """Open two independent grid lots via buy fills; return (strategy, state, ctx)."""
        from core.context import MarketContext
        from core.strategy import Fill
        strategy = self._strategy()
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        state = strategy.get_state("SOL/USD")
        state.grid_lines = [96.0, 98.0, 100.0, 102.0, 104.0]
        state.usdt_per_grid = 20.0
        strategy._buys_allowed = lambda s: True

        for buy_price in (96.0, 98.0):
            cid = str(uuid.uuid4())
            state.orders[cid] = {"side": "buy", "price": buy_price, "qty": 1.0,
                                 "filled": False, "leverage": 1.0}
            strategy.on_fill(Fill(client_id=cid, symbol="SOL/USD", side="buy",
                                  price=buy_price, qty=1.0, fee=0.0, ts=time.time()), ctx)
        return strategy, state, ctx

    def test_two_buys_register_two_lots(self):
        _, _, ctx = self._fill_two_buys()
        assert ctx.open_position_count() == 2
        assert ctx.symbol_position_usdt("SOL/USD") == pytest.approx(96.0 + 98.0)

    def test_sell_fill_removes_only_its_lot(self):
        from core.strategy import Fill
        strategy, state, ctx = self._fill_two_buys()

        # Fill the TP sell belonging to the 96.0 lot only.
        sell_cid = next(cid for cid, o in state.orders.items()
                        if o["side"] == "sell" and o.get("bought_at") == 96.0)
        sell_price = state.orders[sell_cid]["price"]
        strategy.on_fill(Fill(client_id=sell_cid, symbol="SOL/USD", side="sell",
                              price=sell_price, qty=1.0, fee=0.0, ts=time.time()), ctx)

        # The 98.0 lot must still be tracked (bug: cohort wiped -> count 0).
        assert ctx.open_position_count() == 1, \
            "closing one lot must not wipe the whole cohort (#159)"
        remaining = ctx.get_positions("SOL/USD")
        assert len(remaining) == 1 and remaining[0].entry_price == pytest.approx(98.0)

    def test_stop_loss_removes_only_its_lot(self):
        strategy, state, ctx = self._fill_two_buys()

        # Arrange SL levels so ONLY the 96.0 lot stops out at the test price:
        # 96.0 lot SL just below market (fires at 94.0), 98.0 lot SL well below
        # (does not fire) — isolating a single-lot stop.
        for cid, o in state.orders.items():
            if o["side"] != "sell" or "bought_at" not in o:
                continue
            o["sl_price"] = 95.0 if o["bought_at"] == 96.0 else 90.0
        state._direction_score = 0.0  # no momentum-hold delay
        strategy._check_position_stops("SOL/USD", 94.0, state, ctx)

        assert ctx.open_position_count() == 1, \
            "a stop-loss on one lot must not wipe the whole cohort (#159)"
        remaining = ctx.get_positions("SOL/USD")
        assert len(remaining) == 1 and remaining[0].entry_price == pytest.approx(98.0)


# ── Per-coin state persistence across restart (#239) ──────────────────────────

class TestPerCoinStateRestore:
    """The per-coin emergency stop reads state.total_profit, which is rebuilt to
    0.0 on every process start. Without restoring it, a coin that had already
    realized a near-cap loss resumes with its loss counter at 0 and may lose a
    further EMERGENCY_STOP_PCT before the kill switch re-engages (#239). The
    realized loss itself already survives via the persisted cash bucket, so only
    the counter that measures it against the threshold needs restoring.
    """

    def _fresh_strategy(self, monkeypatch):
        from strategies.grid import GridStrategy
        from strategies.grid_params import GridParams
        from core.context import MarketContext
        monkeypatch.setenv("GRIDBOT_BACKTEST", "1")  # skip dashboard logging
        strat = GridStrategy(
            [{"symbol": "SOL/USD", "investment": 100.0, "levels": 5}],
            ml_enabled=False,
            params=GridParams(leverage=1.0),
        )
        strat.init(["SOL/USD"], MarketContext())
        return strat

    def test_init_alone_resets_counter(self, monkeypatch):
        # Baseline: init() rebuilds state fresh — this is exactly the bug surface.
        strat = self._fresh_strategy(monkeypatch)
        assert strat._states["SOL/USD"].total_profit == 0.0

    def test_restore_reinstates_loss_counter(self, monkeypatch):
        strat = self._fresh_strategy(monkeypatch)
        saved = {"SOL/USD": {"investment": 100.0,
                             "total_profit": -11.0, "trade_count": 7}}
        n = strat.restore_paper_state(saved)
        state = strat._states["SOL/USD"]
        assert n == 1
        assert state.total_profit == -11.0
        assert state.trade_count == 7
        assert state.investment == 100.0
        assert state.usdt_per_grid == pytest.approx(100.0 / 5)

    def test_restored_counter_trips_emergency_stop(self, monkeypatch):
        # After restore the engine gate `total_profit <= -(investment*0.12)` must
        # see the persisted near-cap loss. -13 <= -(100*0.12)=-12 → stopped.
        from core.engine import EMERGENCY_STOP_PCT
        strat = self._fresh_strategy(monkeypatch)
        strat.restore_paper_state(
            {"SOL/USD": {"investment": 100.0,
                         "total_profit": -13.0, "trade_count": 4}})
        state = strat._states["SOL/USD"]
        assert state.total_profit <= -(state.investment * EMERGENCY_STOP_PCT)

    def test_restore_does_not_double_compound(self, monkeypatch):
        # Restored `investment` already reflects past compounding; the persisted
        # profit must not be compounded a second time on the next _compound().
        strat = self._fresh_strategy(monkeypatch)
        strat.restore_paper_state(
            {"SOL/USD": {"investment": 150.0,
                         "total_profit": 60.0, "trade_count": 9}})
        state = strat._states["SOL/USD"]
        inv_before = state.investment
        strat._maybe_compound(150.0, state)  # trades_since 9-9=0 → no compound
        assert state.investment == inv_before
        assert state._compounded_profit == 60.0
        assert state._last_compound_at == 9

    def test_restore_ignores_unknown_or_zero_investment(self, monkeypatch):
        strat = self._fresh_strategy(monkeypatch)
        n = strat.restore_paper_state({
            "SOL/USD": {"investment": 0.0, "total_profit": -5.0, "trade_count": 2},
            "ETH/USD": {"investment": 40.0, "total_profit": -3.0, "trade_count": 1},
        })
        # ETH not in _states (not configured) → skipped; SOL has inv<=0 → skipped.
        assert n == 0
        assert strat._states["SOL/USD"].total_profit == 0.0

    def test_restore_empty_is_noop(self, monkeypatch):
        strat = self._fresh_strategy(monkeypatch)
        assert strat.restore_paper_state(None) == 0
        assert strat.restore_paper_state({}) == 0


class TestLoadGridStatesDB:
    """load_grid_states() must round-trip exactly what update_grid_state wrote,
    and skip rows with non-positive investment."""

    def _db(self, monkeypatch, tmp_path):
        from pathlib import Path
        import dashboard.db as db
        monkeypatch.setattr(db, "DB_PATH", Path(tmp_path) / "trades.db")
        return db  # get_conn() auto-creates the schema via _init()

    def test_roundtrip(self, monkeypatch, tmp_path):
        db = self._db(monkeypatch, tmp_path)
        db.update_grid_state(
            "SOL/USD", current_price=100.0, orders={},
            range_pct=0.05, investment=150.0,
            total_profit=-11.5, trade_count=7, prediction="neutral")
        saved = db.load_grid_states()
        assert saved is not None
        assert saved["SOL/USD"]["total_profit"] == pytest.approx(-11.5)
        assert saved["SOL/USD"]["trade_count"] == 7
        assert saved["SOL/USD"]["investment"] == pytest.approx(150.0)

    def test_nonpositive_investment_skipped(self, monkeypatch, tmp_path):
        db = self._db(monkeypatch, tmp_path)
        db.update_grid_state(
            "SOL/USD", current_price=100.0, orders={},
            range_pct=0.05, investment=0.0,
            total_profit=-5.0, trade_count=2, prediction="neutral")
        assert db.load_grid_states() is None

    def test_real_capital_change_clears_grid_state(self, monkeypatch, tmp_path):
        # #239: a real capital change resets the cash buckets (paper_balances=NULL);
        # the per-coin accounting must reset with them so restart does not restore a
        # stale loss counter onto a fresh account.
        db = self._db(monkeypatch, tmp_path)
        db.set_initial_capital(1000.0)
        db.update_grid_state(
            "SOL/USD", current_price=100.0, orders={},
            range_pct=0.05, investment=150.0,
            total_profit=-11.0, trade_count=7, prediction="neutral")
        assert db.load_grid_states() is not None  # persisted while unchanged
        changed = db.set_initial_capital(2000.0)  # real change
        assert changed is True
        assert db.load_grid_states() is None       # per-coin state wiped

    def test_unchanged_capital_keeps_grid_state(self, monkeypatch, tmp_path):
        db = self._db(monkeypatch, tmp_path)
        db.set_initial_capital(1000.0)
        db.update_grid_state(
            "SOL/USD", current_price=100.0, orders={},
            range_pct=0.05, investment=150.0,
            total_profit=-11.0, trade_count=7, prediction="neutral")
        changed = db.set_initial_capital(1000.0)  # same value → no reset
        assert changed is False
        saved = db.load_grid_states()
        assert saved is not None and saved["SOL/USD"]["total_profit"] == pytest.approx(-11.0)
