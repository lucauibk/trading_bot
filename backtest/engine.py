"""
Event-driven backtester using the same Strategy/Broker interface as live trading.
This ensures that backtested strategy behavior matches live behavior exactly.

Usage:
  python -m backtest.engine --strategy grid --symbol SOL/USD --days 90
"""

import argparse
import logging
from typing import List

import pandas as pd

from backtest.metrics import summary
from core.context import MarketContext
from execution.paper import PaperBroker

logger = logging.getLogger("backtest.engine")


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
    """
    broker = PaperBroker(initial_balance=initial_balance)
    ctx = MarketContext()
    ctx.set_equity(initial_balance)

    strategy.init([symbol], ctx)

    pnls: List[float] = []
    equity_curve: List[float] = [initial_balance]
    trade_count = 0

    for i in range(60, len(df)):  # need 60 candles for indicators
        row = df.iloc[i]
        candle_df = df.iloc[max(0, i - 500): i + 1]
        price = float(row["close"])
        ts = row.name  # timestamp

        # Update candle
        strategy.on_candle(symbol, candle_df, ctx)

        # Setup grid on first candle and rebuild every 60 candles (≈ live GRID_REBUILD_CYCLES)
        if hasattr(strategy, "setup_grid") and (i == 60 or (i - 60) % 60 == 0):
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

        # Track equity
        state = strategy.get_state(symbol) if hasattr(strategy, "get_state") else None
        eq = initial_balance + (state.total_profit if state else 0)
        equity_curve.append(eq)

    days = len(df) / 24  # approximate days from hourly candles
    metrics = summary(pnls, equity_curve, days)
    metrics["symbol"] = symbol
    metrics["candles"] = len(df)
    metrics["days"] = round(days, 1)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SOL/USD")
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s")

    import datetime
    from data_fetcher import fetch_ohlcv_since
    # Use paginated Binance fetch to get full history; map /USD → /USDT for Binance
    binance_symbol = args.symbol.replace("/USD", "/USDT").replace("/USDT T", "/USDT")
    since_iso = (datetime.datetime.utcnow() - datetime.timedelta(days=args.days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    df = fetch_ohlcv_since(binance_symbol, "1h", since_iso)
    logger.info("Loaded %d candles for %s", len(df), args.symbol)

    from strategies.grid import GridStrategy
    strategy = GridStrategy(
        [{"symbol": args.symbol, "investment": 200.0, "levels": 8}],
        ml_enabled=False,  # avoid live API calls during backtest
    )

    metrics = run_backtest(strategy, df, args.symbol, initial_balance=200.0)

    print("\n── Backtest Results ──────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<25} {v}")
