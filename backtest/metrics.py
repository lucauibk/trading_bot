"""
Backtest performance metrics.
"""

import math
from typing import List


def sharpe(returns: List[float], periods_per_year: int = 24 * 365) -> float:
    """Annualized Sharpe ratio from list of per-period returns."""
    if len(returns) < 2:
        return 0.0
    import numpy as np
    arr = np.array(returns)
    mean = arr.mean()
    std = arr.std()
    if std < 1e-9:
        return 0.0
    return float((mean / std) * math.sqrt(periods_per_year))


def sortino(returns: List[float], periods_per_year: int = 24 * 365,
            target: float = 0.0) -> float:
    """Annualized Sortino ratio."""
    if len(returns) < 2:
        return 0.0
    import numpy as np
    arr = np.array(returns)
    mean = arr.mean()
    downside = arr[arr < target] - target
    downside_std = math.sqrt((downside ** 2).mean()) if len(downside) > 0 else 1e-9
    return float((mean / downside_std) * math.sqrt(periods_per_year))


def max_drawdown(equity_curve: List[float]) -> float:
    """Maximum drawdown as fraction (e.g. -0.15 = -15%)."""
    if not equity_curve:
        return 0.0
    import numpy as np
    arr = np.array(equity_curve)
    peak = arr.cummax() if hasattr(arr, 'cummax') else arr
    # compute manually
    peak = arr[0]
    max_dd = 0.0
    for v in arr:
        peak = max(peak, v)
        dd = (v - peak) / peak
        max_dd = min(max_dd, dd)
    return float(max_dd)


def calmar(total_return: float, max_dd: float, years: float) -> float:
    """Calmar ratio = annualized_return / |max_drawdown|."""
    if abs(max_dd) < 1e-9 or years <= 0:
        return 0.0
    ann_return = (1 + total_return) ** (1 / years) - 1
    return float(ann_return / abs(max_dd))


def hit_rate(pnls: List[float]) -> float:
    """Fraction of trades that were profitable."""
    if not pnls:
        return 0.0
    wins = sum(1 for p in pnls if p > 0)
    return wins / len(pnls)


def profit_factor(pnls: List[float]) -> float:
    """Gross profit / gross loss."""
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = sum(abs(p) for p in pnls if p < 0)
    if gross_loss < 1e-9:
        return float("inf") if gross_profit > 0 else 1.0
    return gross_profit / gross_loss


def summary(pnls: List[float], equity_curve: List[float], days: float) -> dict:
    """Compute and return all metrics as a dict."""
    # `pnls` is appended per SL/TP tick AND per fill, while `equity_curve` grows one
    # entry per candle (backtest/engine.py) — the two lists have unrelated lengths.
    # Indexing equity_curve[i] with a pnl index therefore (a) can raise IndexError
    # when len(pnls) > len(equity_curve) — silently dropping the whole config from the
    # sweep — and (b) divides each pnl by a chronologically-unrelated candle's equity.
    # Normalize every realized pnl by the starting capital instead: a well-defined,
    # order-independent return series. Sharpe/Sortino are scale-invariant, so the
    # constant denominator does not distort them (#131).
    returns = [p / max(equity_curve[0], 1) for p in pnls] if equity_curve else list(pnls)
    total_pnl = sum(pnls)
    total_return = total_pnl / max(equity_curve[0], 1) if equity_curve else 0

    max_dd = max_drawdown(equity_curve)
    years = days / 365

    return {
        "total_pnl": round(total_pnl, 4),
        "total_return_pct": round(total_return * 100, 2),
        "sharpe": round(sharpe(returns), 3),
        "sortino": round(sortino(returns), 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "calmar": round(calmar(total_return, max_dd, years), 3),
        "hit_rate_pct": round(hit_rate(pnls) * 100, 1),
        "profit_factor": round(profit_factor(pnls), 3),
        "n_trades": len(pnls),
        "avg_win": round(sum(p for p in pnls if p > 0) / max(sum(1 for p in pnls if p > 0), 1), 4),
        "avg_loss": round(sum(p for p in pnls if p < 0) / max(sum(1 for p in pnls if p < 0), 1), 4),
    }
