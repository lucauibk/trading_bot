"""
Directional-Trade OOS-Backtest

Misst die echte Trefferquote der Directional-Trade-Logik auf historischen Daten.
Verwendet das gespeicherte ML-Modell + identische Feature-Extraktion wie im Live-Bot.

Warum:
- OOS-F1 ≈ 0.51 bedeutet bei 2:1 R:R NICHT automatisch +EV:
  Breakeven Win-Rate = 1.5/(3+1.5) = 33.3% (mechanisch bei driftlosem Preis)
- Erst wenn echter Win-Rate auf "up & score≥0.15"-Signalen > ~36% ist Edge nachgewiesen.
- Dieser Test entscheidet ob Directional aktiv bleiben, Gate justiert oder Funding-Arbeit
  priorisiert wird.

Nutzung:
  cd /Users/lucasturz/Projects/trading-bot
  python3 scripts/directional_backtest.py
  python3 scripts/directional_backtest.py --symbol SOL/USD --days 180 --oos-days 90
  python3 scripts/directional_backtest.py --all-symbols
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Projekt-Root zum Suchpfad hinzufügen
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Kein Live-Zugriff auf Dashboard-DB oder Telegram während Backtest
os.environ.setdefault("GRIDBOT_BACKTEST", "1")

import numpy as np
import pandas as pd
import ta as ta_lib

from backtest.data import load_ohlcv
from ml.model import TradingModel
from ml.features.combined import extract_all as extract_all_features

logger = logging.getLogger("directional_backtest")

# ── Konstanten identisch zum Live-Bot (CLAUDE.md) ─────────────────────────
KRAKEN_FEE           = 0.0016   # Maker-Fee 0.16%
TP_ATR_MULT          = 3.0      # Take-Profit: entry + 3×ATR
SL_ATR_MULT          = 1.5      # Stop-Loss:   entry − 1.5×ATR
SCORE_MIN            = 0.15     # directional_score_min aus config/grid_params.json
MAX_HOLD_CANDLES     = 72       # Max. 72h halten, dann Timeout
LEVERAGE             = 2.0      # Konservative Annahme; BREAKEVEN ändert sich mit Leverage nicht
BREAKEVEN_WIN_RATE   = SL_ATR_MULT / (TP_ATR_MULT + SL_ATR_MULT)  # = 1.5/4.5 = 0.333
REQUIRED_EDGE_RATE   = 0.36     # nach Fees + Holding-Cost (konservative Schranke)

SYMBOLS = ["SOL/USD", "ETH/USD", "AVAX/USD", "LINK/USD", "XRP/USD"]


def _get_atr(df: pd.DataFrame, idx: int, period: int = 14) -> float:
    """ATR(14) in absoluten Preis-Einheiten."""
    window = df.iloc[max(0, idx - period): idx + 1]
    if len(window) < 5:
        return float(df["close"].iloc[idx]) * 0.015
    try:
        atr = ta_lib.volatility.average_true_range(
            window["high"], window["low"], window["close"],
            window=min(period, len(window)),
        ).iloc[-1]
        return max(float(atr), float(df["close"].iloc[idx]) * 0.003)
    except Exception:
        return float(df["close"].iloc[idx]) * 0.015


def simulate_trade(df: pd.DataFrame, entry_idx: int, entry_price: float, atr: float) -> dict:
    """
    Simuliert einen Directional-Long-Trade ab entry_idx.
    Gibt Ergebnis-Dict zurück: outcome, pnl_pct, candles_held.
    """
    tp = entry_price + TP_ATR_MULT * atr
    sl = entry_price - SL_ATR_MULT * atr

    for j in range(entry_idx + 1, min(entry_idx + 1 + MAX_HOLD_CANDLES, len(df))):
        h = float(df["high"].iloc[j])
        l = float(df["low"].iloc[j])

        hit_tp = h >= tp
        hit_sl = l <= sl

        if hit_tp and hit_sl:
            # Beide in einer Kerze getroffen: konservativ SL annehmen
            hit_sl = True
            hit_tp = False

        if hit_tp:
            gross = (tp - entry_price) / entry_price * LEVERAGE
            net   = gross - 2 * KRAKEN_FEE * LEVERAGE
            return {"outcome": "win", "pnl_pct": net, "candles": j - entry_idx}

        if hit_sl:
            gross = (sl - entry_price) / entry_price * LEVERAGE
            net   = gross - 2 * KRAKEN_FEE * LEVERAGE
            return {"outcome": "loss", "pnl_pct": net, "candles": j - entry_idx}

    # Timeout: schließen zum letzten verfügbaren Close
    exit_idx  = min(entry_idx + MAX_HOLD_CANDLES, len(df) - 1)
    exit_price = float(df["close"].iloc[exit_idx])
    gross = (exit_price - entry_price) / entry_price * LEVERAGE
    net   = gross - 2 * KRAKEN_FEE * LEVERAGE
    return {"outcome": "timeout", "pnl_pct": net, "candles": MAX_HOLD_CANDLES}


def run_directional_backtest(symbol: str, days: int = 360, oos_days: int = 90) -> dict:
    """
    OOS-Backtest der Directional-Trades für ein Symbol.

    Trainings-/OOS-Split:
    - Letzten `oos_days` Tage = Out-of-Sample (Evaluation)
    - Davor = "Kontext" (nicht für neue Entries genutzt, nur für Feature-Berechnung)
    Das gespeicherte ML-Modell wurde auf 2 Jahren Daten trainiert → OOS-Fenster
    entspricht grob dem letzten Walk-Forward-Fold.
    """
    logger.info("Lade %d Tage OHLCV für %s…", days, symbol)
    df = load_ohlcv(symbol, "1h", days)
    logger.info("%s: %d Candles geladen", symbol, len(df))

    # OOS-Split: letzte oos_days Tage
    oos_candles = oos_days * 24
    if len(df) <= oos_candles + 60:
        logger.warning("%s: zu wenig Daten für OOS-Split", symbol)
        return {}
    oos_start_idx = len(df) - oos_candles

    # Gespeichertes Modell laden
    model = TradingModel(symbol)
    if not model.is_ready():
        logger.warning("%s: kein gespeichertes Modell – übersprungen", symbol)
        return {}
    logger.info("%s: Modell geladen (F1=%.3f, %d Samples)", symbol, model._last_oos_f1, model._n_samples)

    # OOS-Evaluation
    trades = []
    signal_count = 0
    skipped_no_position = 0
    in_trade = False
    trade_end_idx = 0

    for i in range(oos_start_idx, len(df) - 1):
        # Kein überlappender Trade
        if in_trade and i < trade_end_idx:
            continue

        # Features extrahieren (identisch zu _extract_training_features in trainer.py)
        window = df.iloc[max(0, i - 4800): i + 1]
        if len(window) < 60:
            continue

        dt = window.index[-1].to_pydatetime() if hasattr(window.index[-1], "to_pydatetime") else None
        try:
            feats = extract_all_features(window, funding=None, btc=None, btc_corr=0.0, dt=dt)
        except Exception as exc:
            logger.debug("Feature-Extraktion fehlgeschlagen %s i=%d: %s", symbol, i, exc)
            continue

        # ML-Vorhersage
        label_int, lgbm_conf, lgbm_dir_score = model.predict(feats)
        signal_count += 1

        # Directional-Gate: identisch zu strategies/grid.py:760 (prediction=="up" & score≥0.15)
        # dir_score = buy_prob - sell_prob; >0.15 = "up" Signal
        if lgbm_dir_score < SCORE_MIN:
            continue

        # Einstieg simulieren
        entry_price = float(df["close"].iloc[i])
        atr = _get_atr(df, i)

        result = simulate_trade(df, i, entry_price, atr)
        result["idx"]    = i
        result["ts"]     = str(df.index[i])[:16]
        result["score"]  = round(lgbm_dir_score, 3)
        result["conf"]   = round(lgbm_conf, 3)
        result["entry"]  = round(entry_price, 4)
        result["atr"]    = round(atr, 4)
        result["tp"]     = round(entry_price + TP_ATR_MULT * atr, 4)
        result["sl"]     = round(entry_price - SL_ATR_MULT * atr, 4)
        trades.append(result)

        # Nächsten Trade frühestens nach aktuellem Trade-Ende
        in_trade = True
        trade_end_idx = i + result["candles"] + 1

    logger.info("%s: %d Signale geprüft, %d Trades ausgelöst", symbol, signal_count, len(trades))

    if not trades:
        return {
            "symbol": symbol,
            "model_f1": model._last_oos_f1,
            "oos_days": oos_days,
            "n_signals_checked": signal_count,
            "n_trades": 0,
            "verdict": "⚠️  KEIN SIGNAL — zu wenig Directional-Trades für Aussage",
        }

    # Metriken
    wins    = [t for t in trades if t["outcome"] == "win"]
    losses  = [t for t in trades if t["outcome"] == "loss"]
    timeouts = [t for t in trades if t["outcome"] == "timeout"]
    n       = len(trades)
    win_rate = len(wins) / n

    pnls = [t["pnl_pct"] for t in trades]
    total_pnl_pct = sum(pnls)
    avg_pnl_pct   = total_pnl_pct / n
    avg_win_pct   = sum(t["pnl_pct"] for t in wins) / max(len(wins), 1)
    avg_loss_pct  = sum(t["pnl_pct"] for t in losses) / max(len(losses), 1)
    avg_hold      = sum(t["candles"] for t in trades) / n

    if win_rate >= REQUIRED_EDGE_RATE:
        verdict = f"✅  ECHTER EDGE — Win-Rate {win_rate:.1%} > {REQUIRED_EDGE_RATE:.1%} Schwelle → Directional aktiv lassen"
    elif win_rate >= BREAKEVEN_WIN_RATE + 0.01:
        verdict = f"⚠️  KNAPP — Win-Rate {win_rate:.1%} > {BREAKEVEN_WIN_RATE:.1%} Breakeven aber < {REQUIRED_EDGE_RATE:.1%} Schwelle → nach Fees negativ"
    else:
        verdict = f"❌  KEIN EDGE — Win-Rate {win_rate:.1%} ≤ {BREAKEVEN_WIN_RATE:.1%} Breakeven → Directional kostet Geld"

    return {
        "symbol":          symbol,
        "model_f1":        round(model._last_oos_f1, 3),
        "oos_days":        oos_days,
        "n_signals_checked": signal_count,
        "n_trades":        n,
        "n_wins":          len(wins),
        "n_losses":        len(losses),
        "n_timeouts":      len(timeouts),
        "win_rate":        round(win_rate, 3),
        "breakeven":       round(BREAKEVEN_WIN_RATE, 3),
        "required_edge":   REQUIRED_EDGE_RATE,
        "total_pnl_pct":   round(total_pnl_pct * 100, 2),
        "avg_pnl_pct":     round(avg_pnl_pct * 100, 3),
        "avg_win_pct":     round(avg_win_pct * 100, 3),
        "avg_loss_pct":    round(avg_loss_pct * 100, 3),
        "avg_hold_candles": round(avg_hold, 1),
        "verdict":         verdict,
        "trades":          trades,
    }


def print_results(r: dict):
    print(f"\n{'═' * 60}")
    print(f"  {r.get('symbol', '?')}  |  OOS {r.get('oos_days', '?')} Tage  |  Modell F1={r.get('model_f1', '?')}")
    print(f"{'═' * 60}")

    if r.get("n_trades", 0) == 0:
        print(f"  {r.get('verdict', '')}")
        print(f"  Signale geprüft:  {r.get('n_signals_checked', 0)}")
        return

    print(f"  Signale geprüft:  {r['n_signals_checked']}")
    print(f"  Trades ausgelöst: {r['n_trades']}")
    print(f"  Wins / Losses / Timeouts:  {r['n_wins']} / {r['n_losses']} / {r['n_timeouts']}")
    print(f"  Win-Rate:  {r['win_rate']:.1%}   (Breakeven: {r['breakeven']:.1%}, Schwelle: {r['required_edge']:.1%})")
    print(f"  Ø Hold:    {r['avg_hold_candles']:.0f} Kerzen")
    print(f"  Ø Win:     {r['avg_win_pct']:+.2f}%  |  Ø Loss: {r['avg_loss_pct']:+.2f}%")
    print(f"  Gesamt-PnL (simuliert, {LEVERAGE:.0f}x Lev.): {r['total_pnl_pct']:+.2f}%")
    print(f"\n  {r['verdict']}")
    print()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    )

    parser = argparse.ArgumentParser(description="Directional-Trade OOS-Backtest")
    parser.add_argument("--symbol", default=None, help="Einzelnes Symbol (z.B. SOL/USD)")
    parser.add_argument("--all-symbols", action="store_true", help="Alle konfigurierten Symbole")
    parser.add_argument("--days",     type=int, default=360, help="Gesamte Historie (Tage)")
    parser.add_argument("--oos-days", type=int, default=90,  help="OOS-Fenster (Tage)")
    parser.add_argument("--verbose",  action="store_true", help="Einzelne Trades ausgeben")
    args = parser.parse_args()

    symbols = SYMBOLS if args.all_symbols else [args.symbol or "SOL/USD"]

    all_trades = []
    for sym in symbols:
        r = run_directional_backtest(sym, days=args.days, oos_days=args.oos_days)
        if r:
            print_results(r)
            all_trades.extend(r.get("trades", []))

    # Gesamt-Zusammenfassung über alle Symbole
    if len(symbols) > 1 and all_trades:
        n    = len(all_trades)
        wins = sum(1 for t in all_trades if t["outcome"] == "win")
        wr   = wins / n
        total_pnl = sum(t["pnl_pct"] for t in all_trades) * 100

        print(f"\n{'═' * 60}")
        print(f"  ALLE SYMBOLE KOMBINIERT")
        print(f"{'═' * 60}")
        print(f"  Trades:    {n}  |  Win-Rate: {wr:.1%}  (Breakeven: {BREAKEVEN_WIN_RATE:.1%})")
        print(f"  Gesamt-PnL (simuliert, {LEVERAGE:.0f}x): {total_pnl:+.2f}%")

        if wr >= REQUIRED_EDGE_RATE:
            print(f"\n  ✅  ECHTER EDGE ÜBER ALLE SYMBOLE")
            print(f"  → Schritt 2 (Funding-Backfill) priorisieren für weiteres Upside")
            print(f"  → Directional aktiv lassen, Leverage/Größe ggf. erhöhen")
        elif wr > BREAKEVEN_WIN_RATE:
            print(f"\n  ⚠️  KNAPPER EDGE — Vor Skalierung Funding-Backfill durchführen")
        else:
            print(f"\n  ❌  KEIN EDGE — Directional ausschalten bis Funding-Signal vorhanden")
            print(f"  → Schritt 2 (Funding-Backfill) ist jetzt die einzige Möglichkeit Edge zu erzeugen")
        print()


if __name__ == "__main__":
    main()
