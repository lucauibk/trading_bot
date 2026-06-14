"""
Event-driven backtester using the same Strategy/Broker interface as live trading.
This ensures that backtested strategy behavior matches live behavior exactly.

Usage:
  python -m backtest.engine --symbol SOL/USD --days 90
  python -m backtest.engine --symbol SOL/USD --days 180 --params '{"sl_mode": "floor"}'
"""

import argparse
import json
import logging
import os
from typing import List

import pandas as pd

# Backtests must never write to the live dashboard DB or ping Telegram
os.environ.setdefault("GRIDBOT_BACKTEST", "1")

from backtest.metrics import summary
from core.context import MarketContext
from core.engine import EMERGENCY_STOP_PCT
from execution.paper import PaperBroker

logger = logging.getLogger("backtest.engine")

REBUILD_EVERY = 60  # candles, ≈ live GRID_REBUILD_CYCLES


def run_backtest(
    strategy,
    df: pd.DataFrame,
    symbol: str,
    initial_balance: float = 1000.0,
) -> dict:
    """
    Run a backtest on historical OHLCV data.

    strategy: a Strategy instance (already initialized)
    df: OHLCV DataFrame with datetime index
    symbol: trading pair
    initial_balance: starting USDT balance

    Returns the metrics dict; "equity_curve" and "pnls" keys carry the raw
    series for downstream analysis (sweep/OOS).
    """
    broker = PaperBroker(initial_balance=initial_balance)
    ctx = MarketContext()
    ctx.set_equity(initial_balance)

    strategy.init([symbol], ctx)

    pnls: List[float] = []
    equity_curve: List[float] = [initial_balance]
    trade_count = 0
    halted = False

    for i in range(60, len(df)):  # need 60 candles for indicators
        row = df.iloc[i]
        candle_df = df.iloc[max(0, i - 500): i + 1]
        price = float(row["close"])

        state = strategy.get_state(symbol) if hasattr(strategy, "get_state") else None

        # Emergency-stop parity with live engine (core/engine.py): symbol halts
        # for good once realized loss exceeds the kill-switch threshold.
        if state and state.total_profit <= -(state.investment * EMERGENCY_STOP_PCT):
            halted = True
            break

        # Update candle
        strategy.on_candle(symbol, candle_df, ctx)

        # Out-of-range rebuild parity with live engine (price escaped grid ±1%)
        out_of_range = False
        if state and state.grid_lines:
            lo, hi = state.grid_lines[0], state.grid_lines[-1]
            out_of_range = price < lo * 0.99 or price > hi * 1.01

        # Setup grid on first candle, periodically, or when out of range
        if hasattr(strategy, "setup_grid") and (
            i == 60 or (i - 60) % REBUILD_EVERY == 0 or out_of_range
        ):
            strategy.setup_grid(symbol, price, ctx)

        # Process tick (SL/TP checks, trailing) — capture SL/TP pnl
        state_before_tick = getattr(strategy.get_state(symbol) if hasattr(strategy, "get_state") else None, "total_profit", 0)
        strategy.on_tick(symbol, price, ctx)
        state_after_tick = getattr(strategy.get_state(symbol) if hasattr(strategy, "get_state") else None, "total_profit", 0)
        tick_pnl = state_after_tick - state_before_tick
        if abs(tick_pnl) > 1e-6:
            pnls.append(tick_pnl)
            trade_count += 1

        # Process paper broker fills (grid buy/sell)
        fills = broker.update_price(symbol, price)
        for fill in fills:
            prev_profit = getattr(strategy.get_state(symbol), "total_profit", 0)
            strategy.on_fill(fill, ctx)
            new_profit = getattr(strategy.get_state(symbol), "total_profit", 0)
            trade_pnl = new_profit - prev_profit
            if abs(trade_pnl) > 1e-6:
                pnls.append(trade_pnl)
                trade_count += 1

        # Sync desired orders
        desired = {o.client_id: o for o in strategy.desired_orders(symbol, price, ctx) if o.client_id}
        active_ids = {o.client_id for o in broker.get_open_orders(symbol)}
        for cid in list(active_ids):
            if cid not in desired:
                broker.cancel(cid)
        for cid, order in desired.items():
            if cid not in active_ids:
                broker.place_limit(
                    symbol=order.symbol, side=order.side,
                    price=order.price, qty=order.qty,
                    post_only=order.post_only, client_id=cid,
                )

        # Track equity including unrealized PnL of open positions — without it,
        # drawdown during long underwater holds is invisible.
        state = strategy.get_state(symbol) if hasattr(strategy, "get_state") else None
        eq = initial_balance + (state.total_profit if state else 0)
        if state:
            for o in state.orders.values():
                if (o.get("side") == "sell" and not o.get("filled")
                        and "bought_at" in o and not o.get("pre_seeded")):
                    eq += (price - o["bought_at"]) * o["qty"]
            if state._directional:
                d = state._directional
                eq += (price - d["entry"]) * d["qty"]
        equity_curve.append(eq)

    days = len(df) / 24  # approximate days from hourly candles
    metrics = summary(pnls, equity_curve, days)
    metrics["symbol"] = symbol
    metrics["candles"] = len(df)
    metrics["days"] = round(days, 1)
    metrics["halted"] = halted
    metrics["equity_curve"] = equity_curve
    metrics["pnls"] = pnls
    return metrics


def load_data(symbol: str, days: int) -> pd.DataFrame:
    from backtest.data import load_ohlcv
    return load_ohlcv(symbol, "1h", days)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SOL/USD")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--balance", type=float, default=200.0)
    parser.add_argument("--params", default=None, help="GridParams overrides as JSON string")
    parser.add_argument("--params-file", default=None, help="Path to JSON file with GridParams overrides")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s")

    df = load_data(args.symbol, args.days)
    logger.info("Loaded %d candles for %s", len(df), args.symbol)

    from strategies.grid import GridStrategy
    from strategies.grid_params import GridParams

    overrides = {}
    if args.params_file:
        with open(args.params_file) as f:
            overrides = json.load(f)
    if args.params:
        overrides.update(json.loads(args.params))
    params = GridParams.from_dict(overrides)

    strategy = GridStrategy(
        [{"symbol": args.symbol, "investment": args.balance, "levels": 8}],
        ml_enabled=False,  # avoid live API calls during backtest
        params=params,
    )

    metrics = run_backtest(strategy, df, args.symbol, initial_balance=args.balance)
    metrics.pop("equity_curve", None)
    metrics.pop("pnls", None)

    print("\n── Backtest Results ──────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<25} {v}")
