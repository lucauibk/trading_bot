"""
Grid Optimizer – findet die besten Parameter pro Symbol und Markt-Regime.

Aufruf:
  python3 grid_optimizer.py                    # alle Symbole, alle Regime
  python3 grid_optimizer.py --symbol SOL/USD   # einzelnes Symbol
  python3 grid_optimizer.py --since 2024-06-01T00:00:00Z

Ausgabe:
  - Top-5 Kombinationen pro Symbol und Regime
  - REGIME_CONFIGS Dict direkt zum Einfügen in grid_bot.py
"""

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd
import ta

sys.path.insert(0, str(Path(__file__).parent))
from data_fetcher import fetch_ohlcv_since

# ── Konfiguration ──────────────────────────────────────────────────────────────
SYMBOLS    = ["SOL/USD", "ETH/USD", "DOT/USD", "LINK/USD", "DOGE/USD"]
SINCE      = "2024-01-01T00:00:00Z"
TIMEFRAME  = "1h"
INVESTMENT = 300.0
FEE        = 0.0016   # Kraken Maker-Fee ~0.16%

# Parameter-Raster (vollständiger Sweep)
LEVELS_OPTIONS = [4, 6, 8, 10, 12, 14]
RANGE_OPTIONS  = [0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20]

# Regime-Schwellen (müssen mit price_predictor/indicators.py übereinstimmen)
ADX_TRENDING_THRESHOLD  = 25.0
ATR_PCT_VOLATILE_THRESHOLD = 3.0

TOP_N = 5  # Wie viele Top-Kombinationen pro Kategorie anzeigen


# ── Regime-Tagging ─────────────────────────────────────────────────────────────

def tag_regimes(df: pd.DataFrame) -> pd.Series:
    """
    Taggt jeden Candle mit dem Markt-Regime:
      'trending'  → ADX > 25
      'volatile'  → ATR% > 3% (und nicht trending)
      'ranging'   → sonst
    """
    adx = ta.trend.ADXIndicator(
        df["high"], df["low"], df["close"], window=14
    ).adx().fillna(0)

    atr = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=14
    ).average_true_range().fillna(0)

    atr_pct = atr / df["close"].replace(0, float("nan")) * 100

    regime = pd.Series("ranging", index=df.index)
    regime[adx > ADX_TRENDING_THRESHOLD] = "trending"
    regime[(adx <= ADX_TRENDING_THRESHOLD) & (atr_pct > ATR_PCT_VOLATILE_THRESHOLD)] = "volatile"
    return regime


# ── Grid-Backtest ──────────────────────────────────────────────────────────────

def backtest_grid(df: pd.DataFrame, investment: float, levels: int,
                  range_pct: float, fee: float = FEE) -> dict:
    """
    Exakte Replikation der PaperGridBot-Logik:
    - Half-Step-Offset
    - Echte Buy/Sell-Paare
    - with_position=True (obere Hälfte vorbelegt)
    - Grid-Reset wenn Preis ±1% außerhalb der Range
    """
    if len(df) < 2:
        return _empty_result(investment)

    usdt_per_grid = investment / levels
    total_profit  = 0.0
    trades        = 0
    resets        = 0
    peak_profit   = 0.0
    max_drawdown  = 0.0

    i = 0
    while i < len(df):
        start_price = df.iloc[i]["close"]
        lower = start_price * (1 - range_pct)
        upper = start_price * (1 + range_pct)
        step  = (upper - lower) / levels
        grid_lines = [lower + (k + 0.5) * step for k in range(levels)]

        orders = {}
        for k, price in enumerate(grid_lines):
            qty = usdt_per_grid / price
            if price < start_price:
                orders[price] = {"side": "buy", "qty": qty, "filled": False}
            else:
                buy_price = grid_lines[k - 1] if k > 0 else start_price
                orders[price] = {
                    "side": "sell",
                    "qty": usdt_per_grid / buy_price,
                    "filled": False,
                    "bought_at": buy_price,
                }

        j = i + 1
        while j < len(df):
            row  = df.iloc[j]
            high = row["high"]
            low  = row["low"]

            for price, order in list(orders.items()):
                if order["filled"]:
                    continue

                if order["side"] == "buy" and low <= price:
                    order["filled"] = True
                    idx = grid_lines.index(price)
                    if idx < len(grid_lines) - 1:
                        sell_price = grid_lines[idx + 1]
                        orders[sell_price] = {
                            "side": "sell",
                            "qty": order["qty"],
                            "filled": False,
                            "bought_at": price,
                        }

                elif order["side"] == "sell" and high >= price and "bought_at" in order:
                    buy_price  = order["bought_at"]
                    qty        = order["qty"]
                    profit     = (price - buy_price) * qty
                    trade_fee  = (price + buy_price) * qty * fee
                    net        = profit - trade_fee
                    total_profit += net
                    trades += 1
                    order["filled"] = True
                    orders[buy_price] = {
                        "side": "buy",
                        "qty": usdt_per_grid / buy_price,
                        "filled": False,
                    }

                    if total_profit > peak_profit:
                        peak_profit = total_profit
                    dd = (peak_profit - total_profit) / investment if peak_profit > 0 else 0
                    if dd > max_drawdown:
                        max_drawdown = dd

            if low < lower * 0.99 or high > upper * 1.01:
                resets += 1
                i = j
                break
            j += 1
        else:
            break

    days         = max((df.index[-1] - df.index[0]).days, 1)
    daily_profit = total_profit / days
    return_pct   = total_profit / investment * 100
    avg_per_trade = total_profit / trades if trades > 0 else 0

    # Score: gewichtet Ertrag, Handelsfrequenz und Drawdown
    score = (daily_profit * 0.5
             + return_pct * 0.3
             - max_drawdown * investment * 0.2)

    return {
        "levels":      levels,
        "range_pct":   range_pct,
        "trades":      trades,
        "profit":      total_profit,
        "daily":       daily_profit,
        "monthly":     daily_profit * 30,
        "return_pct":  return_pct,
        "avg_trade":   avg_per_trade,
        "max_dd_pct":  max_drawdown * 100,
        "resets":      resets,
        "days":        days,
        "score":       score,
    }


def _empty_result(investment: float) -> dict:
    return {k: 0 for k in
            ["levels","range_pct","trades","profit","daily","monthly",
             "return_pct","avg_trade","max_dd_pct","resets","days","score"]}


# ── Optimierung ────────────────────────────────────────────────────────────────

def optimize_all(df: pd.DataFrame, investment: float) -> list:
    """Testet alle Kombinationen auf dem gesamten DataFrame."""
    results = []
    for levels, range_pct in itertools.product(LEVELS_OPTIONS, RANGE_OPTIONS):
        r = backtest_grid(df, investment, levels, range_pct)
        r["levels"] = levels
        r["range_pct"] = range_pct
        results.append(r)
    return sorted(results, key=lambda x: x["score"], reverse=True)


def optimize_per_regime(df: pd.DataFrame, investment: float) -> dict:
    """
    Testet alle Kombinationen getrennt pro Markt-Regime.
    Gibt {regime: [sorted_results]} zurück.
    """
    regimes = tag_regimes(df)
    regime_dfs = {
        "ranging":  df[regimes == "ranging"],
        "trending": df[regimes == "trending"],
        "volatile": df[regimes == "volatile"],
    }

    per_regime = {}
    for regime, rdf in regime_dfs.items():
        coverage = len(rdf) / len(df) * 100
        if len(rdf) < 100:
            per_regime[regime] = []
            continue

        results = []
        for levels, range_pct in itertools.product(LEVELS_OPTIONS, RANGE_OPTIONS):
            r = backtest_grid(rdf, investment, levels, range_pct)
            r["levels"] = levels
            r["range_pct"] = range_pct
            r["coverage_pct"] = coverage
            results.append(r)

        per_regime[regime] = sorted(results, key=lambda x: x["score"], reverse=True)

    return per_regime, {k: len(v) for k, v in regime_dfs.items()}


# ── Ausgabe ────────────────────────────────────────────────────────────────────

def _print_header(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def _print_table(results: list, top_n: int = TOP_N):
    if not results:
        print("  (Nicht genug Daten)")
        return
    header = f"  {'Lvl':<5} {'Range%':<8} {'Trades':<8} {'Profit':<9} {'Ø/Tag':<8} {'Ø/Mon':<9} {'Return':<8} {'MaxDD':<8} {'Score'}"
    print(header)
    print(f"  {'─'*85}")
    for r in results[:top_n]:
        print(
            f"  {r['levels']:<5} {r['range_pct']*100:<8.0f} "
            f"{r['trades']:<8} {r['profit']:<9.2f} "
            f"{r['daily']:<8.3f} {r['monthly']:<9.2f} "
            f"{r['return_pct']:<8.1f} {r['max_dd_pct']:<8.1f} "
            f"{r['score']:.3f}"
        )


def _regime_config_line(regime: str, best: dict) -> str:
    return (
        f'    "{regime}": {{"levels": {best["levels"]}, '
        f'"range_pct": {best["range_pct"]}}},  '
        f'# Ø {best["daily"]:.3f} USDT/Tag | {best["return_pct"]:.0f}% Return'
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Grid Parameter Optimizer")
    parser.add_argument("--symbol", help="Einzelnes Symbol (z.B. SOL/USD)")
    parser.add_argument("--since",  default=SINCE, help="Start-Datum ISO 8601")
    parser.add_argument("--investment", type=float, default=INVESTMENT, help="USDT pro Coin")
    args = parser.parse_args()

    symbols    = [args.symbol] if args.symbol else SYMBOLS
    investment = args.investment
    combos     = len(LEVELS_OPTIONS) * len(RANGE_OPTIONS)

    print("=" * 60)
    print(f"  GRID OPTIMIZER  |  {args.since[:10]} – heute")
    print(f"  {len(symbols)} Coin(s)  |  {combos} Kombinationen  |  Fee: {FEE*100:.2f}%")
    print("=" * 60)

    # Gesamt-beste pro Symbol + Regime-Daten sammeln
    symbol_bests    = []
    all_regime_best = {"ranging": [], "trending": [], "volatile": []}

    for symbol in symbols:
        print(f"\n▶ {symbol}  – Lade Daten…", end=" ", flush=True)
        try:
            df = fetch_ohlcv_since(symbol, TIMEFRAME, args.since)
        except Exception as e:
            print(f"FEHLER: {e}")
            continue

        if len(df) < 200:
            print(f"Zu wenig Daten ({len(df)} Candles), übersprungen.")
            continue

        print(f"{len(df)} Candles ({(df.index[-1]-df.index[0]).days} Tage)")

        # ── Gesamt-Optimierung ─────────────────────────────────────────────
        _print_header(f"{symbol} – Alle Regime (Top {TOP_N})")
        all_results = optimize_all(df, investment)
        _print_table(all_results)
        best = all_results[0]
        print(f"\n  ★ Beste: {best['levels']} Levels | ±{best['range_pct']*100:.0f}% | "
              f"Ø {best['daily']:.3f} USDT/Tag | {best['return_pct']:.0f}% Return | "
              f"MaxDD {best['max_dd_pct']:.1f}%")
        symbol_bests.append({**best, "symbol": symbol})

        # ── Regime-Optimierung ─────────────────────────────────────────────
        per_regime, regime_sizes = optimize_per_regime(df, investment)

        regime_labels = {
            "ranging":  f"Ranging  ({regime_sizes['ranging']:>5} Candles = {regime_sizes['ranging']/len(df)*100:.0f}%)",
            "trending": f"Trending ({regime_sizes['trending']:>5} Candles = {regime_sizes['trending']/len(df)*100:.0f}%)",
            "volatile": f"Volatile ({regime_sizes['volatile']:>5} Candles = {regime_sizes['volatile']/len(df)*100:.0f}%)",
        }
        for regime in ("ranging", "trending", "volatile"):
            results = per_regime[regime]
            _print_header(f"{symbol} – {regime_labels[regime]}")
            _print_table(results)
            if results:
                b = results[0]
                all_regime_best[regime].append({**b, "symbol": symbol})

    # ── Gesamt-Empfehlung ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  GESAMT-RANKING  –  BESTE COIN + PARAMETER")
    print("=" * 60)
    symbol_bests.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(symbol_bests, 1):
        print(f"  {i}. {r['symbol']:<12} {r['levels']} Levels | ±{r['range_pct']*100:.0f}% | "
              f"Ø {r['daily']:.3f}/Tag | {r['return_pct']:.0f}% Return")

    # ── Regime-Empfehlung ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  REGIME-CONFIGS  –  In grid_bot.py eintragen:")
    print("=" * 60)
    print("\nREGIME_CONFIGS = {")
    for regime in ("ranging", "trending", "volatile"):
        bests = all_regime_best[regime]
        if not bests:
            fallback = {"ranging": (8, 0.05), "trending": (6, 0.15), "volatile": (10, 0.10)}
            lvl, rng = fallback[regime]
            print(f'    "{regime}": {{"levels": {lvl}, "range_pct": {rng}}},  # (Fallback – zu wenig Daten)')
            continue

        # Median über alle Symbole wählen (robuster als einzelner Best)
        bests.sort(key=lambda x: x["score"], reverse=True)
        levels_votes  = sorted(b["levels"]    for b in bests)
        range_votes   = sorted(b["range_pct"] for b in bests)
        median_levels = levels_votes[len(levels_votes) // 2]
        median_range  = range_votes[len(range_votes) // 2]
        avg_daily     = sum(b["daily"] for b in bests) / len(bests)
        avg_return    = sum(b["return_pct"] for b in bests) / len(bests)
        print(f'    "{regime}": {{"levels": {median_levels}, "range_pct": {median_range}}},  '
              f'# Ø {avg_daily:.3f} USDT/Tag | {avg_return:.0f}% Return')
    print("}")

    if symbol_bests:
        best = symbol_bests[0]
        print(f"\n  → GRIDS Eintrag für besten Coin:")
        print(f'  {{"symbol": "{best["symbol"]}", "investment": {investment:.0f}, "levels": {best["levels"]}}}')


if __name__ == "__main__":
    main()
