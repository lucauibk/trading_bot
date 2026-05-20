#!/usr/bin/env python3
"""
Trading-Optimizer CLI — analysiert historische Trade-Daten und leitet
Verbesserungen für Grid-Params, ML-Confidence und Risiko-Filter ab.

Verwendung:
  python scripts/optimize.py --analyze-trades [--symbol SOL/USD] [--days 30]
  python scripts/optimize.py --calibration-report [--symbol SOL/USD]
  python scripts/optimize.py --suggest-params [--symbol SOL/USD]
  python scripts/optimize.py --pattern-mine [--days 60]
  python scripts/optimize.py --run-sweep --symbol SOL/USD
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

DB_TRADES = ROOT / "data" / "trades.db"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _conn(db_path: Path = DB_TRADES) -> sqlite3.Connection:
    if not db_path.exists():
        print(f"[FEHLER] DB nicht gefunden: {db_path}")
        sys.exit(1)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def _trades_df(symbol: str = None, days: int = 30, con=None) -> pd.DataFrame:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    sql = "SELECT t.*, tc.* FROM trades t LEFT JOIN trade_context tc ON tc.trade_id = t.id WHERE t.timestamp >= ?"
    params = [cutoff]
    if symbol:
        sql += " AND t.symbol = ?"
        params.append(symbol)
    df = pd.read_sql(sql, con, params=params, parse_dates=["timestamp"])
    return df


def _divider(title: str = ""):
    w = 60
    if title:
        pad = (w - len(title) - 2) // 2
        print("=" * pad + f" {title} " + "=" * (w - pad - len(title) - 2))
    else:
        print("=" * w)


# ── --analyze-trades ──────────────────────────────────────────────────────────

def cmd_analyze_trades(symbol: str, days: int):
    _divider(f"Trade-Analyse | {days}d | {symbol or 'alle Coins'}")
    con = _conn()
    df = _trades_df(symbol, days, con)
    con.close()

    if df.empty:
        print("Keine Trades in diesem Zeitraum.")
        return

    total = len(df)
    wins  = (df["pnl"] > 0).sum()
    pnl_total = df["pnl"].sum()
    print(f"Trades gesamt : {total}")
    print(f"Win-Rate      : {wins/total*100:.1f}%  ({wins}/{total})")
    print(f"Gesamt-PnL    : {pnl_total:+.4f} USDT")
    print(f"Avg PnL/Trade : {df['pnl'].mean():+.4f} USDT")
    print(f"Bester Trade  : {df['pnl'].max():+.4f} USDT")
    print(f"Schlechtester : {df['pnl'].min():+.4f} USDT")

    # Win-Rate pro Regime
    if "regime" in df.columns and df["regime"].notna().any():
        print()
        _divider("Win-Rate pro Regime")
        for regime, grp in df.groupby("regime"):
            if not regime:
                continue
            wr = (grp["pnl"] > 0).mean() * 100
            avg = grp["pnl"].mean()
            print(f"  {regime:<12} | Trades: {len(grp):>4} | WR: {wr:>5.1f}% | Avg: {avg:+.4f} USDT")

    # Win-Rate pro ML-Confidence-Bucket
    if "ml_confidence" in df.columns and df["ml_confidence"].notna().any():
        print()
        _divider("Win-Rate pro ML-Confidence-Bucket")
        df["conf_bucket"] = pd.cut(df["ml_confidence"], bins=[0, 0.55, 0.6, 0.65, 0.7, 0.8, 1.0])
        for bucket, grp in df.groupby("conf_bucket"):
            if grp.empty:
                continue
            wr = (grp["pnl"] > 0).mean() * 100
            avg = grp["pnl"].mean()
            print(f"  Conf {str(bucket):<22} | Trades: {len(grp):>4} | WR: {wr:>5.1f}% | Avg: {avg:+.4f} USDT")

    # Verlustreichste Stunden
    if "timestamp" in df.columns:
        print()
        _divider("Top-3 verlustreichste Stunden")
        df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
        hourly = df.groupby("hour")["pnl"].mean().sort_values()
        for h, avg_pnl in hourly.head(3).items():
            print(f"  {h:02d}:00 UTC | Avg PnL: {avg_pnl:+.4f} USDT")

    # Holding-Time Vergleich
    if "holding_seconds" in df.columns and df["holding_seconds"].notna().any():
        print()
        _divider("Holding-Time: Gewinn vs. Verlust")
        wins_ht  = df[df["pnl"] > 0]["holding_seconds"].mean() / 3600
        loss_ht  = df[df["pnl"] < 0]["holding_seconds"].mean() / 3600
        print(f"  Gewinn-Trades : ø {wins_ht:.1f}h Haltezeit")
        print(f"  Verlust-Trades: ø {loss_ht:.1f}h Haltezeit")

    _divider()


# ── --calibration-report ──────────────────────────────────────────────────────

def cmd_calibration_report(symbol: str):
    _divider(f"ML-Kalibrierungsbericht | {symbol or 'alle Coins'}")
    con = _conn()
    sql = "SELECT * FROM predictions WHERE hit IS NOT NULL"
    params = []
    if symbol:
        sql += " AND symbol = ?"
        params.append(symbol)
    df = pd.read_sql(sql, con, params=params)
    con.close()

    if df.empty:
        print("Keine kalibrierten Vorhersagen vorhanden.")
        print("Tipp: realisierte Preise werden automatisch nachgetragen sobald 6h vergangen sind.")
        return

    print(f"Kalibrierte Vorhersagen: {len(df)}")

    # Brier-Score
    if "hit" in df.columns and "confidence" in df.columns:
        hit_arr  = df["hit"].astype(float).values
        conf_arr = df["confidence"].astype(float).values
        brier = float(np.mean((conf_arr - hit_arr) ** 2))
        print(f"Brier-Score: {brier:.4f}  (0=perfekt, 0.25=zufällig)")

    # Reliability pro Bucket
    print()
    _divider("Reliability-Tabelle (Konfidenz-Bucket vs. tatsächliche Hit-Rate)")
    df["bucket"] = pd.cut(df["confidence"], bins=[0, 0.55, 0.6, 0.65, 0.7, 0.8, 1.0])
    print(f"  {'Bucket':<26} | {'N':>5} | {'Hit-Rate':>9} | {'Conf (avg)':>10}")
    for bucket, grp in df.groupby("bucket"):
        if grp.empty:
            continue
        hit_rate = grp["hit"].mean() * 100
        conf_avg = grp["confidence"].mean()
        print(f"  {str(bucket):<26} | {len(grp):>5} | {hit_rate:>8.1f}% | {conf_avg:>10.3f}")

    # Empfehlung MIN_CONFIDENCE
    best_bucket = None
    best_f1 = 0.0
    for bucket, grp in df.groupby("bucket"):
        if len(grp) < 10:
            continue
        precision = grp["hit"].mean()
        recall = len(grp) / len(df)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_bucket = bucket

    if best_bucket:
        print()
        print(f"[EMPFEHLUNG] MIN_CONFIDENCE = {best_bucket.left:.2f}  (bester F1={best_f1:.3f})")
        print(f"  → In ml/predictor.py: MIN_CONFIDENCE = {best_bucket.left:.2f}")

    _divider()


# ── --suggest-params ──────────────────────────────────────────────────────────

def cmd_suggest_params(symbol: str):
    _divider(f"Parameter-Empfehlung | {symbol or 'alle Coins'}")
    con = _conn()

    # Beste Backtest-Runs aus optimizer_runs
    sql = "SELECT * FROM optimizer_runs"
    params = []
    if symbol:
        sql += " WHERE symbol = ?"
        params.append(symbol)
    df_runs = pd.read_sql(sql, con, params=params)

    if df_runs.empty:
        print("Keine Optimizer-Runs in DB. Führe erst aus:")
        print("  python scripts/optimize.py --run-sweep --symbol SOL/USD")
        con.close()
        return

    print(f"Optimizer-Runs: {len(df_runs)}")
    print()
    _divider("Beste Parameter pro Regime")
    print(f"  {'Regime':<12} | {'Levels':>6} | {'Range%':>7} | {'Score':>7} | {'Daily%':>7} | {'MaxDD%':>7}")
    for regime in ["ranging", "trending", "volatile", ""]:
        grp = df_runs[df_runs["regime"] == regime] if regime else df_runs[df_runs["regime"].isin(["", None])]
        if grp.empty:
            continue
        best = grp.loc[grp["score"].idxmax()]
        params_dict = json.loads(best["params_json"])
        regime_label = regime or "all"
        print(
            f"  {regime_label:<12} | {params_dict.get('levels', '-'):>6} "
            f"| {params_dict.get('range_pct', 0)*100:>6.1f}% "
            f"| {best['score']:>7.4f} | {best['daily_pct']*100:>6.2f}% "
            f"| {best['max_dd']*100:>6.2f}%"
        )

    print()
    _divider("YAML-Diff für strategy_params.yaml")
    for regime in ["ranging", "trending", "volatile"]:
        grp = df_runs[df_runs["regime"] == regime]
        if grp.empty:
            continue
        best = grp.loc[grp["score"].idxmax()]
        p = json.loads(best["params_json"])
        print(f"  {regime}:")
        print(f"    levels: {p.get('levels', '?')}")
        print(f"    range_pct: {p.get('range_pct', 0):.3f}")

    con.close()
    _divider()


# ── --pattern-mine ────────────────────────────────────────────────────────────

def cmd_pattern_mine(days: int):
    _divider(f"Pattern-Mining (letzte {days}d) – toxische Trade-Setups")
    con = _conn()
    df = _trades_df(symbol=None, days=days, con=con)
    con.close()

    if df.empty or "atr_pct" not in df.columns:
        print("Nicht genug Kontext-Daten für Pattern-Mining.")
        print("Tipp: stelle sicher dass trade_context-Spalten befüllt sind (grid_bot.py).")
        return

    losses = df[df["pnl"] < 0].copy()
    if losses.empty:
        print("Keine Verlust-Trades – Glückwunsch!")
        return

    print(f"Verlust-Trades analysiert: {len(losses)}")
    print()

    # Einfaches regel-basiertes Clustering (statt DBSCAN um scipy-Abhängigkeit zu vermeiden)
    rules = []

    def _check_rule(condition_mask, label: str, condition_str: str):
        sub = losses[condition_mask]
        if len(sub) < 3:
            return
        avg_pnl = sub["pnl"].mean()
        pct_of_losses = len(sub) / len(losses) * 100
        rules.append((avg_pnl, len(sub), pct_of_losses, label, condition_str))

    if "atr_pct" in losses.columns and losses["atr_pct"].notna().any():
        _check_rule(losses["atr_pct"] > 0.04, "Hohe Volatilität",
                    "atr_pct > 4% → Overtrading bei extremer Volatilität")
        _check_rule(losses["atr_pct"] < 0.008, "Sehr niedrige Volatilität",
                    "atr_pct < 0.8% → Range zu eng für Fee-Abdeckung")

    if "rsi" in losses.columns and losses["rsi"].notna().any():
        _check_rule(losses["rsi"] < 25, "Extreme Überverkauft",
                    "RSI < 25 → möglicherweise Capitulation (Bag-Hold)")
        _check_rule(losses["rsi"] > 75, "Extreme Überkauft",
                    "RSI > 75 → Grid in Overbought-Zone aufgebaut")

    if "regime" in losses.columns and losses["regime"].notna().any():
        _check_rule(losses["regime"] == "volatile", "Volatile Regime",
                    "Regime=volatile → häufig Out-of-Range Resets")
        _check_rule(losses["regime"] == "trending", "Trending Regime",
                    "Regime=trending → Grid läuft gegen Trend")

    if "ml_confidence" in losses.columns and losses["ml_confidence"].notna().any():
        _check_rule(losses["ml_confidence"] < 0.55, "Niedrige ML-Konfidenz",
                    "ml_confidence < 0.55 → regelbasierter Fallback oft falsch")

    if not rules:
        print("Keine signifikanten Muster gefunden (zu wenig Daten oder kein Kontext).")
        return

    rules.sort(key=lambda r: r[0])  # nach avg_pnl aufsteigend
    print(f"  {'Muster':<25} | {'N':>4} | {'Anteil':>7} | {'Avg Loss':>10} | Beschreibung")
    for avg_pnl, n, pct, label, desc in rules:
        print(f"  {label:<25} | {n:>4} | {pct:>6.1f}% | {avg_pnl:>+10.4f} | {desc}")

    print()
    print("[EMPFEHLUNGEN]")
    for avg_pnl, n, pct, label, desc in rules[:3]:
        if "atr_pct > 4%" in desc:
            print("  → Füge in predict_direction() hinzu: if atr_pct > 0.04: return 'neutral'")
        elif "RSI < 25" in desc:
            print("  → Erhöhe PER_POS_SL_PCT auf 0.06 wenn RSI < 25 und regime=volatile")
        elif "ml_confidence < 0.55" in desc:
            print("  → Setze MIN_CONFIDENCE = 0.58 in ml/predictor.py")
    _divider()


# ── --run-sweep ───────────────────────────────────────────────────────────────

def cmd_run_sweep(symbol: str):
    _divider(f"Grid-Parameter-Sweep | {symbol}")
    sys.path.insert(0, str(ROOT))

    try:
        import grid_optimizer as go
    except Exception as e:
        print(f"[FEHLER] grid_optimizer.py nicht importierbar: {e}")
        return

    try:
        from data_fetcher import fetch_ohlcv
        df = fetch_ohlcv(symbol, "1h", 500)
    except Exception as e:
        print(f"[FEHLER] Daten für {symbol} nicht verfügbar: {e}")
        return

    try:
        from dashboard.db import log_optimizer_run
    except Exception as e:
        print(f"[FEHLER] DB nicht verfügbar: {e}")
        log_optimizer_run = None

    print(f"Sweep über {len(go.LEVELS_OPTIONS)} × {len(go.RANGE_OPTIONS)} Kombinationen…")
    results = go.run_sweep(symbol, df)

    saved = 0
    for r in results:
        print(
            f"  Levels={r['levels']:>2} Range={r['range_pct']*100:.1f}% "
            f"Regime={r.get('regime','all'):<10} "
            f"Score={r['score']:.4f} Daily={r['daily_pct']*100:.2f}% MaxDD={r['max_dd']*100:.2f}%"
        )
        if log_optimizer_run:
            try:
                log_optimizer_run(
                    symbol=symbol,
                    regime=r.get("regime", ""),
                    params={"levels": r["levels"], "range_pct": r["range_pct"]},
                    score=r["score"],
                    daily_pct=r["daily_pct"],
                    max_dd=r["max_dd"],
                    sample_size=r.get("sample_size", len(df)),
                )
                saved += 1
            except Exception as ex:
                print(f"  [WARN] DB-Speicher fehlgeschlagen: {ex}")

    print(f"\n{len(results)} Sweep-Ergebnisse, {saved} in DB gespeichert.")
    _divider()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Grid Trading Bot – Optimizer CLI")
    parser.add_argument("--analyze-trades",    action="store_true")
    parser.add_argument("--calibration-report", action="store_true")
    parser.add_argument("--suggest-params",    action="store_true")
    parser.add_argument("--pattern-mine",      action="store_true")
    parser.add_argument("--run-sweep",         action="store_true")
    parser.add_argument("--symbol",  type=str, default=None, help="z.B. SOL/USD")
    parser.add_argument("--days",    type=int, default=30,   help="Analyse-Zeitraum in Tagen")
    args = parser.parse_args()

    if args.analyze_trades:
        cmd_analyze_trades(args.symbol, args.days)
    elif args.calibration_report:
        cmd_calibration_report(args.symbol)
    elif args.suggest_params:
        cmd_suggest_params(args.symbol)
    elif args.pattern_mine:
        cmd_pattern_mine(args.days)
    elif args.run_sweep:
        if not args.symbol:
            print("[FEHLER] --run-sweep benötigt --symbol, z.B. --symbol SOL/USD")
            sys.exit(1)
        cmd_run_sweep(args.symbol)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
