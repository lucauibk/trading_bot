"""
Grid Backtester – simuliert die aktuelle Grid-Bot-Logik auf historischen Daten.
Aufruf: python3 grid_backtester.py

Verwendet dieselbe Grid-Logik wie grid_bot.py:
- Half-Step-Offset (Preis immer zwischen zwei Levels)
- Echte Buy/Sell-Paare (nur Profit wenn wirklich gekauft wurde)
- with_position=True: obere Hälfte vorbelegt (wie bei UP-Vorhersage)
- Alle 5 Coins aus der aktuellen GRIDS-Konfiguration
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from data_fetcher import fetch_ohlcv_since

# ── Konfiguration (spiegelt grid_bot.py) ─────────────────────────────────────
GRIDS = [
    {"symbol": "SOL/USD",  "investment": 300.0, "levels": 8},
    {"symbol": "LINK/USD", "investment": 300.0, "levels": 8},
    {"symbol": "DOT/USD",  "investment": 300.0, "levels": 8},
    {"symbol": "ETH/USD",  "investment": 300.0, "levels": 8},
    {"symbol": "DOGE/USD", "investment": 300.0, "levels": 8},
]

SINCE         = "2024-01-01T00:00:00Z"
TIMEFRAME     = "1h"
RANGE_PCT     = 0.05   # feste Range für Backtest (ATR nicht verfügbar auf historisch)
FEE           = 0.001  # 0.1% pro Trade (Kraken Maker ~0.16%, Taker ~0.26%)
WITH_POSITION = True   # True = UP-Vorhersage simulieren


def run_grid_backtest(df: pd.DataFrame, investment: float, levels: int,
                      range_pct: float, with_position: bool = True) -> dict:
    usdt_per_grid = investment / levels
    total_profit  = 0.0
    trades        = 0
    resets        = 0

    i = 0
    while i < len(df):
        start_price = df.iloc[i]["close"]
        lower = start_price * (1 - range_pct)
        upper = start_price * (1 + range_pct)
        step  = (upper - lower) / levels

        # Half-Step-Offset: Preis liegt immer zwischen zwei Levels
        grid_lines = [lower + (k + 0.5) * step for k in range(levels)]

        # Orders initialisieren wie setup_grid()
        orders = {}
        for k, price in enumerate(grid_lines):
            qty = usdt_per_grid / price
            if price < start_price:
                orders[price] = {"side": "buy", "qty": qty, "filled": False}
            elif with_position:
                buy_price = grid_lines[k - 1] if k > 0 else start_price
                orders[price] = {
                    "side": "sell",
                    "qty": usdt_per_grid / buy_price,
                    "filled": False,
                    "bought_at": buy_price,
                }
            else:
                orders[price] = {"side": "sell", "qty": qty, "filled": False}

        j = i + 1
        while j < len(df):
            row  = df.iloc[j]
            high = row["high"]
            low  = row["low"]

            for price, order in list(orders.items()):
                if order["filled"]:
                    continue

                # BUY: Preis fällt auf oder unter Grid-Level
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

                # SELL: nur wenn wirklich vorher gekauft
                elif order["side"] == "sell" and high >= price and "bought_at" in order:
                    buy_price  = order["bought_at"]
                    qty        = order["qty"]
                    profit     = (price - buy_price) * qty
                    fee        = (price + buy_price) * qty * FEE
                    total_profit += profit - fee
                    trades     += 1
                    order["filled"] = True
                    # Kauf-Order zurücksetzen
                    orders[buy_price] = {
                        "side": "buy",
                        "qty": usdt_per_grid / buy_price,
                        "filled": False,
                    }

            # Grid neu aufbauen wenn aus Range
            if low < lower * 0.99 or high > upper * 1.01:
                resets += 1
                i = j
                break

            j += 1
        else:
            break

    days = max((df.index[-1] - df.index[0]).days, 1)
    return {
        "trades":       trades,
        "total_profit": total_profit,
        "daily_avg":    total_profit / days,
        "monthly_avg":  total_profit / days * 30,
        "return_pct":   total_profit / investment * 100,
        "grid_resets":  resets,
        "days":         days,
    }


if __name__ == "__main__":
    print("=" * 55)
    print(f"  GRID BACKTESTER | {SINCE[:10]} – heute")
    print(f"  Range: ±{RANGE_PCT*100:.0f}% | Levels: {GRIDS[0]['levels']} | "
          f"with_position={WITH_POSITION}")
    print("=" * 55)

    total_investment = sum(g["investment"] for g in GRIDS)
    summary_profit   = 0.0

    for g in GRIDS:
        symbol = g["symbol"]
        print(f"\n[{symbol}] Lade Daten…", end=" ", flush=True)
        try:
            df = fetch_ohlcv_since(symbol, TIMEFRAME, SINCE)
        except Exception as e:
            print(f"FEHLER: {e}")
            continue

        print(f"{len(df)} Candles")
        r = run_grid_backtest(df, g["investment"], g["levels"], RANGE_PCT, WITH_POSITION)
        summary_profit += r["total_profit"]

        sign = "+" if r["total_profit"] >= 0 else ""
        print(f"  Zeitraum:       {r['days']} Tage")
        print(f"  Grid Resets:    {r['grid_resets']}")
        print(f"  Trades gesamt:  {r['trades']}")
        print(f"  Gesamt Profit:  {sign}{r['total_profit']:.2f} USDT")
        print(f"  Ø Profit/Tag:   {sign}{r['daily_avg']:.2f} USDT")
        print(f"  Ø Profit/Monat: {sign}{r['monthly_avg']:.2f} USDT")
        print(f"  Return:         {sign}{r['return_pct']:.1f}%")

    print("\n" + "=" * 55)
    sign = "+" if summary_profit >= 0 else ""
    print(f"  ALLE COINS | Investment: {total_investment:.0f} USDT")
    print(f"  Gesamt Profit:  {sign}{summary_profit:.2f} USDT")
    print(f"  Ø Monat:        {sign}{summary_profit / (df.index[-1] - df.index[0]).days * 30:.2f} USDT")
    print(f"  Return gesamt:  {sign}{summary_profit / total_investment * 100:.1f}%")
    print("=" * 55)
