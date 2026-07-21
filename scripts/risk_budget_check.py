#!/usr/bin/env python3
"""
Dollar-risk audit for per_pos_sl_max_pct and floor_sl_atr_mult (Phase 2 of the
Ultra-Bot improvement plan). Read-only, no writes — analysis only.

Motivation: per_pos_sl_max_pct (4%) and floor_sl_atr_mult (1.0) were never
checked against absolute dollar-risk scenarios, only against percentage caps.
This script grounds the check in the bot's ACTUAL live state (data/trades.db
bot_status/coin_settings), not sweep.py's synthetic INVESTMENT=200/LEVERAGE=3
placeholder, since the two can diverge (dashboard per-coin overrides reduce
some symbols' real investment well below the flat per-coin split).

Verwendung:
  python3 scripts/risk_budget_check.py
"""

import sqlite3
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from strategies.grid import MAX_INVESTMENT_MULT
from strategies.grid_params import GridParams

DB_TRADES = ROOT / "data" / "trades.db"
CONFIG_YAML = ROOT / "config" / "config.yaml"
GRID_PARAMS_JSON = ROOT / "config" / "grid_params.json"

ATR_LOOKBACK_DAYS = 90


def _conn() -> sqlite3.Connection:
    if not DB_TRADES.exists():
        print(f"[FEHLER] DB nicht gefunden: {DB_TRADES}")
        sys.exit(1)
    con = sqlite3.connect(DB_TRADES)
    con.row_factory = sqlite3.Row
    return con


def load_live_state() -> dict:
    con = _conn()
    try:
        bot_status = dict(con.execute("SELECT * FROM bot_status WHERE id=1").fetchone())
        coin_settings = {
            row["symbol"]: dict(row)
            for row in con.execute("SELECT * FROM coin_settings")
        }
    finally:
        con.close()
    return {"bot_status": bot_status, "coin_settings": coin_settings}


def load_grid_params() -> GridParams:
    import json
    if GRID_PARAMS_JSON.exists():
        overrides = json.loads(GRID_PARAMS_JSON.read_text())
    else:
        overrides = {}
    return GridParams.from_dict(overrides)


def load_risk_budget() -> dict:
    cfg = yaml.safe_load(CONFIG_YAML.read_text())
    return cfg.get("risk", {})


def per_symbol_investment(bot_status: dict, coin_settings: dict) -> dict:
    """Replicate main.py's investment logic: dashboard max_investment can only
    REDUCE the flat per-coin split, never raise it above it."""
    enabled = [s for s, r in coin_settings.items() if r.get("enabled")]
    initial_capital = bot_status.get("initial_capital") or 0.0
    if not enabled or initial_capital <= 0:
        return {}
    per_coin = initial_capital / len(enabled)
    result = {}
    for sym in enabled:
        override = coin_settings[sym].get("max_investment")
        initial_investment = min(override, per_coin) if override else per_coin
        result[sym] = {
            "initial_investment": round(initial_investment, 2),
            "compounded_max": round(initial_investment * MAX_INVESTMENT_MULT, 2),
        }
    return result


def worst_case_per_position_loss(investment: float, levels: int, leverage: float,
                                   per_pos_sl_max_pct: float) -> float:
    """worst_case = (investment / levels) x leverage x per_pos_sl_max_pct — a
    single position's SL-hit dollar loss, per strategies/grid.py's
    usdt_per_grid = investment/levels and qty already baking in leverage."""
    usdt_per_grid = investment / max(levels, 1)
    return usdt_per_grid * leverage * per_pos_sl_max_pct


def estimate_floor_sl_loss(investment: float, leverage: float, atr_pct: float,
                            floor_sl_atr_mult: float) -> float:
    """Coarse approximation for the floor-SL scenario (only relevant if
    sl_mode="floor" — NOT the current live mode, see main output). Assumes
    worst case: the full grid investment is deployed and unwound at the
    floor, floor_distance_pct ~= floor_sl_atr_mult x atr_pct."""
    floor_distance_pct = floor_sl_atr_mult * atr_pct / 100
    return investment * leverage * floor_distance_pct


def fetch_atr_pct(symbol: str) -> float:
    try:
        from backtest.data import load_ohlcv
        from price_predictor.indicators import compute_indicators
        df = load_ohlcv(symbol, "1h", ATR_LOOKBACK_DAYS)
        df = compute_indicators(df)
        return float(df["atr_pct"].dropna().mean())
    except Exception as e:
        print(f"  [WARN] ATR-Fetch fehlgeschlagen fuer {symbol}: {e}")
        return float("nan")


def ampel(loss: float, budget: float) -> str:
    if budget <= 0:
        return "?"
    ratio = loss / budget
    if ratio < 0.5:
        return "OK"
    if ratio < 1.0:
        return "WARN"
    return "OVER"


def main():
    state = load_live_state()
    bot_status = state["bot_status"]
    coin_settings = state["coin_settings"]
    params = load_grid_params()
    risk_budget = load_risk_budget()

    leverage = bot_status.get("leverage") or 1.0
    initial_capital = bot_status.get("initial_capital") or 0.0
    max_risk_per_trade_usd = risk_budget.get("max_risk_per_trade", 0.01) * initial_capital
    max_portfolio_risk_usd = risk_budget.get("max_portfolio_risk", 0.05) * initial_capital

    print("=" * 78)
    print("Risk-Budget-Check — per_pos_sl_max_pct / floor_sl_atr_mult")
    print("=" * 78)
    print(f"Live sl_mode (config/grid_params.json): {params.sl_mode}")
    print(f"Live leverage (bot_status): {leverage}x")
    print(f"Live initial_capital (bot_status): {initial_capital} USDT")
    if CONFIG_YAML.exists():
        yaml_capital = yaml.safe_load(CONFIG_YAML.read_text()).get("initial_capital")
        if yaml_capital and yaml_capital != initial_capital:
            print(f"  [HINWEIS] config.yaml initial_capital={yaml_capital} weicht vom "
                  f"live bot_status.initial_capital={initial_capital} ab — "
                  f"Budget-Rechnung unten nutzt den LIVE-Wert.")
    print(f"Budget max_risk_per_trade: {max_risk_per_trade_usd:.2f} USDT "
          f"({risk_budget.get('max_risk_per_trade', 0.01):.0%} von initial_capital)")
    print(f"Budget max_portfolio_risk: {max_portfolio_risk_usd:.2f} USDT "
          f"({risk_budget.get('max_portfolio_risk', 0.05):.0%} von initial_capital)")
    regime_caps = {r: params.sl_max_pct_for_regime(r) for r in params.regime_levels}
    print(f"per_pos_sl_max_pct (default): {params.per_pos_sl_max_pct:.1%}  |  "
          f"per-regime: {', '.join(f'{r}={c:.1%}' for r, c in regime_caps.items())}  |  "
          f"floor_sl_atr_mult: {params.floor_sl_atr_mult}  |  "
          f"MAX_INVESTMENT_MULT: {MAX_INVESTMENT_MULT}x")
    print()

    investments = per_symbol_investment(bot_status, coin_settings)
    if not investments:
        print("[FEHLER] Keine aktiven Symbole/kein initial_capital in bot_status gefunden.")
        sys.exit(1)

    print("-" * 78)
    print("Szenario A: per_pos_sl_max_pct (aktueller Live-Modus: per_position)")
    print("-" * 78)
    header = f"{'Symbol':<10}{'Regime':<10}{'Invest.':>10}{'Zustand':<12}{'$-Verlust':>12}{'Ampel':>8}"
    print(header)
    worst = {"loss": 0.0, "symbol": None, "regime": None, "state": None}
    for sym, inv in investments.items():
        for regime, levels in params.regime_levels.items():
            for state_label, investment in (
                ("initial", inv["initial_investment"]),
                ("compounded", inv["compounded_max"]),
            ):
                loss = worst_case_per_position_loss(
                    investment, levels, leverage, params.sl_max_pct_for_regime(regime))
                flag = ampel(loss, max_risk_per_trade_usd)
                print(f"{sym:<10}{regime:<10}{investment:>10.2f}{state_label:<12}{loss:>12.2f}{flag:>8}")
                if loss > worst["loss"]:
                    worst = {"loss": loss, "symbol": sym, "regime": regime, "state": state_label}

    print()
    print(f"Schlechtester Fall: {worst['symbol']} / {worst['regime']} / {worst['state']} "
          f"= {worst['loss']:.2f} USDT ({worst['loss']/max_risk_per_trade_usd:.0%} des "
          f"max_risk_per_trade-Budgets von {max_risk_per_trade_usd:.2f} USDT)")

    trigger = worst["loss"] > max_risk_per_trade_usd
    print()
    if trigger:
        print("[TRIGGER] Mindestens ein Szenario ueberschreitet das max_risk_per_trade-Budget "
              "-> konkreter Aenderungsvorschlag noetig (siehe Plan Phase 2), keine automatische "
              "Aenderung durch dieses Skript.")
    else:
        print("[BESTAETIGUNG] Alle Szenarien (inkl. kompoundiert) liegen innerhalb des "
              "max_risk_per_trade-Budgets bei den aktuellen per-regime-Caps "
              f"({', '.join(f'{r}={c:.1%}' for r, c in regime_caps.items())}). "
              "Keine weitere Aenderung noetig.")

    print()
    print("-" * 78)
    print("Szenario B: floor_sl_atr_mult (informativ — sl_mode ist aktuell NICHT 'floor' live)")
    print("-" * 78)
    print(f"{'Symbol':<10}{'ATR%(90d)':>12}{'Invest.':>10}{'Zustand':<12}{'$-Verlust (approx.)':>20}")
    for sym, inv in investments.items():
        atr_pct = fetch_atr_pct(sym)
        for state_label, investment in (
            ("initial", inv["initial_investment"]),
            ("compounded", inv["compounded_max"]),
        ):
            if atr_pct != atr_pct:  # NaN check
                print(f"{sym:<10}{'n/a':>12}{investment:>10.2f}{state_label:<12}{'n/a':>20}")
                continue
            loss = estimate_floor_sl_loss(investment, leverage, atr_pct, params.floor_sl_atr_mult)
            print(f"{sym:<10}{atr_pct:>11.2f}%{investment:>10.2f}{state_label:<12}{loss:>20.2f}")
    print()
    print("Hinweis: Szenario B ist eine grobe Naeherung (voller Grid-Invest zum Floor-Abstand "
          "unwound) und nur relevant, falls sl_mode jemals auf 'floor' zurueckgestellt wird.")


if __name__ == "__main__":
    main()
