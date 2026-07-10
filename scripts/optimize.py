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


# ── --ready-for-live ─────────────────────────────────────────────────────────

_GO     = "  [GO]     "
_CAUTION= "  [CAUTION]"
_NOGO   = "  [NO-GO]  "

MIN_TRADES        = 50    # Mindest-Trade-Anzahl
MIN_PAPER_DAYS    = 7     # Mindest-Paper-Trading-Dauer
MIN_WIN_RATE      = 0.50  # 50% Win-Rate
MAX_DRAWDOWN      = 0.15  # 15% Max-Drawdown
MAX_BRIER         = 0.22  # Brier-Score-Grenze
MIN_ML_SAMPLES    = 20    # Mindest-kalibrierte Predictions
MAX_ERROR_RATE    = 0.02  # Max 2% ERROR-Zeilen im Log
MAX_CRASH_ERRORS  = 3     # Kritische Fehler/Tracebacks maximal


def _parse_logs_for_errors(log_path, days: int = 7) -> dict:
    """Liest Log-File und zählt ERRORs/Tracebacks der letzten N Tage."""
    result = {"total_lines": 0, "error_lines": 0, "critical_lines": 0,
              "traceback_count": 0, "crash_markers": [], "log_days": 0,
              "first_ts": None, "last_ts": None}

    if not log_path.exists():
        return result

    cutoff = datetime.now() - timedelta(days=days)
    in_traceback = False
    traceback_buf = []
    import re
    ts_re = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")

    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = ts_re.match(line)
            if m:
                try:
                    ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    continue
                if result["first_ts"] is None:
                    result["first_ts"] = ts
                result["last_ts"] = ts
                if ts < cutoff:
                    in_traceback = False
                    continue
                result["total_lines"] += 1
                if "[ERROR]" in line or "[CRITICAL]" in line:
                    result["error_lines"] += 1
                    if "[CRITICAL]" in line:
                        result["critical_lines"] += 1
                if "Traceback" in line:
                    in_traceback = True
                    traceback_buf = [line.rstrip()]
                elif in_traceback:
                    if line.startswith(" ") or line.startswith("\t"):
                        traceback_buf.append(line.rstrip())
                    else:
                        result["traceback_count"] += 1
                        result["crash_markers"].append(" | ".join(traceback_buf[-2:]))
                        in_traceback = False
                        traceback_buf = []
            else:
                if in_traceback and (line.startswith(" ") or line.startswith("\t")):
                    traceback_buf.append(line.rstrip())

    if result["first_ts"] and result["last_ts"]:
        result["log_days"] = (result["last_ts"] - result["first_ts"]).total_seconds() / 86400
    return result


def cmd_ready_for_live(days: int):
    _divider("READY FOR LIVE – Bereitschafts-Check")
    print(f"  Analyse-Zeitraum: letzte {days} Tage | Grenzwerte: "
          f"WR≥{MIN_WIN_RATE*100:.0f}% | MaxDD≤{MAX_DRAWDOWN*100:.0f}% | "
          f"Brier≤{MAX_BRIER} | Trades≥{MIN_TRADES}")
    print()

    checks = []   # (status, label, detail, fix)
    warnings = []

    # ── 1. Trade-Daten laden ──────────────────────────────────────────────────
    con = _conn()
    df = _trades_df(symbol=None, days=days, con=con)

    # ── 2. Mindest-Trade-Anzahl ───────────────────────────────────────────────
    n_trades = len(df)
    if n_trades >= MIN_TRADES:
        checks.append((_GO, "Sample-Size",
                        f"{n_trades} Trades (≥{MIN_TRADES} benötigt)", None))
    elif n_trades >= MIN_TRADES // 2:
        checks.append((_CAUTION, "Sample-Size",
                        f"Nur {n_trades} Trades (≥{MIN_TRADES} benötigt)",
                        f"Weiter Paper-traden bis {MIN_TRADES}+ Trades erreicht"))
    else:
        checks.append((_NOGO, "Sample-Size",
                        f"Nur {n_trades} Trades – zu wenig für Aussage (≥{MIN_TRADES} benötigt)",
                        f"Mindestens {MIN_TRADES - n_trades} weitere Trades sammeln"))

    if df.empty:
        print("  [KEINE DATEN] Keine Trades in DB – Live-Betrieb nicht möglich.")
        con.close()
        return

    # ── 3. Paper-Trading-Dauer ────────────────────────────────────────────────
    first_trade = pd.to_datetime(df["timestamp"]).min()
    last_trade  = pd.to_datetime(df["timestamp"]).max()
    paper_days  = (last_trade - first_trade).total_seconds() / 86400
    if paper_days >= MIN_PAPER_DAYS:
        checks.append((_GO, "Paper-Dauer",
                        f"{paper_days:.1f} Tage Paper-Trading (≥{MIN_PAPER_DAYS} benötigt)", None))
    else:
        checks.append((_NOGO, "Paper-Dauer",
                        f"Nur {paper_days:.1f} Tage Paper-Trading",
                        f"Noch {MIN_PAPER_DAYS - paper_days:.0f} weitere Tage abwarten"))

    # ── 4. Win-Rate & PnL ────────────────────────────────────────────────────
    win_rate = (df["pnl"] > 0).mean()
    total_pnl = df["pnl"].sum()
    avg_pnl   = df["pnl"].mean()
    if win_rate >= MIN_WIN_RATE and total_pnl > 0:
        checks.append((_GO, "Win-Rate / PnL",
                        f"WR={win_rate*100:.1f}% | Total={total_pnl:+.2f} USDT | Avg={avg_pnl:+.4f} USDT", None))
    elif win_rate >= MIN_WIN_RATE - 0.05 or total_pnl > 0:
        detail = f"WR={win_rate*100:.1f}% | Total={total_pnl:+.2f} USDT | Avg={avg_pnl:+.4f} USDT"
        fix = []
        if win_rate < MIN_WIN_RATE:
            fix.append(f"Win-Rate noch {(MIN_WIN_RATE - win_rate)*100:.1f}% unter Ziel")
        if total_pnl <= 0:
            fix.append("Gesamt-PnL negativ")
        checks.append((_CAUTION, "Win-Rate / PnL", detail, " | ".join(fix)))
    else:
        checks.append((_NOGO, "Win-Rate / PnL",
                        f"WR={win_rate*100:.1f}% (Ziel ≥{MIN_WIN_RATE*100:.0f}%) | "
                        f"Total={total_pnl:+.2f} USDT",
                        "Strategie-Parameter prüfen (/optimizer Win-Rate-Modus)"))

    # ── 5. Trend letzte 7 Tage ───────────────────────────────────────────────
    recent_cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    df_recent = df[pd.to_datetime(df["timestamp"], utc=True) >= recent_cutoff]
    if not df_recent.empty:
        recent_pnl = df_recent["pnl"].sum()
        recent_wr  = (df_recent["pnl"] > 0).mean()
        if recent_pnl > 0 and recent_wr >= MIN_WIN_RATE:
            checks.append((_GO, "Trend (7d)",
                            f"Letzte 7d: +{recent_pnl:.2f} USDT | WR={recent_wr*100:.1f}%", None))
        elif recent_pnl > 0 or recent_wr >= MIN_WIN_RATE - 0.05:
            checks.append((_CAUTION, "Trend (7d)",
                            f"Letzte 7d: {recent_pnl:+.2f} USDT | WR={recent_wr*100:.1f}%",
                            "Performance noch nicht stabil"))
        else:
            checks.append((_NOGO, "Trend (7d)",
                            f"Letzte 7d: {recent_pnl:+.2f} USDT | WR={recent_wr*100:.1f}%",
                            "Aktuell negativ – Bot gerade nicht profitabel"))
    else:
        warnings.append("Keine Trades in den letzten 7 Tagen – Trend-Check nicht möglich")

    # ── 6. Max Drawdown aus equity-Tabelle ───────────────────────────────────
    try:
        # The equity table's value column is `capital` (dashboard/db.py:34-38),
        # not `equity`. Selecting `equity` raised OperationalError, which the
        # except-branch below silently turned into a warning — so the Max-Drawdown
        # live-readiness gate never actually evaluated (#102).
        eq_df = pd.read_sql(
            "SELECT timestamp, capital FROM equity ORDER BY timestamp",
            con, parse_dates=["timestamp"]
        )
        if not eq_df.empty and len(eq_df) > 5:
            peak = eq_df["capital"].expanding().max()
            drawdown = ((eq_df["capital"] - peak) / peak).min()
            abs_dd   = abs(float(drawdown))
            if abs_dd <= MAX_DRAWDOWN:
                checks.append((_GO, "Max Drawdown",
                                f"Max DD = {abs_dd*100:.1f}% (Limit: {MAX_DRAWDOWN*100:.0f}%)", None))
            elif abs_dd <= MAX_DRAWDOWN * 1.5:
                checks.append((_CAUTION, "Max Drawdown",
                                f"Max DD = {abs_dd*100:.1f}% (Limit: {MAX_DRAWDOWN*100:.0f}%)",
                                "Drawdown nahe Grenze – Risiko-Parameter prüfen"))
            else:
                checks.append((_NOGO, "Max Drawdown",
                                f"Max DD = {abs_dd*100:.1f}% – zu hoch (Limit: {MAX_DRAWDOWN*100:.0f}%)",
                                "PER_POS_SL_PCT / MAX_LOSS_PCT senken; /optimizer Lose-Rate prüfen"))
        else:
            warnings.append("Equity-Tabelle leer oder zu wenig Punkte – Drawdown nicht berechenbar")
    except Exception as exc:
        # Surface the actual error (e.g. a schema/column mismatch) instead of a
        # generic message, so a broken drawdown gate can't hide silently again.
        warnings.append(f"Equity-Tabelle nicht auswertbar: {type(exc).__name__}: {exc}")

    # ── 7. Regime-Abdeckung ───────────────────────────────────────────────────
    if "regime" in df.columns and df["regime"].notna().any():
        regimes_seen = df["regime"].dropna().unique().tolist()
        n_regimes    = len(regimes_seen)
        if n_regimes >= 2:
            checks.append((_GO, "Regime-Abdeckung",
                            f"{n_regimes} Regimes getestet: {', '.join(regimes_seen)}", None))
        else:
            checks.append((_CAUTION, "Regime-Abdeckung",
                            f"Nur Regime '{regimes_seen[0]}' gesehen – Bot nicht in allen Marktphasen getestet",
                            "Warten bis trending + volatile + ranging alle aufgetreten sind"))
    else:
        warnings.append("Kein regime-Kontext in Trades – trade_context-Logging prüfen")

    # ── 8. ML-Kalibrierung ───────────────────────────────────────────────────
    try:
        ml_df = pd.read_sql(
            "SELECT confidence, hit FROM predictions WHERE hit IS NOT NULL",
            con
        )
        if len(ml_df) >= MIN_ML_SAMPLES:
            brier = float(np.mean((ml_df["confidence"].astype(float) - ml_df["hit"].astype(float)) ** 2))
            if brier <= MAX_BRIER:
                checks.append((_GO, "ML-Kalibrierung",
                                f"Brier-Score={brier:.4f} (≤{MAX_BRIER} = gut) | "
                                f"{len(ml_df)} kalibrierte Predictions", None))
            elif brier <= MAX_BRIER * 1.3:
                checks.append((_CAUTION, "ML-Kalibrierung",
                                f"Brier-Score={brier:.4f} – leicht erhöht",
                                "MIN_CONFIDENCE anheben (--calibration-report zeigt optimalen Wert)"))
            else:
                checks.append((_NOGO, "ML-Kalibrierung",
                                f"Brier-Score={brier:.4f} – schlecht (nahe Zufallsrate 0.25)",
                                "ML-Modell hat schlechte Kalibrierung – Retrain abwarten oder "
                                "MIN_CONFIDENCE stark anheben"))
        else:
            n_missing = MIN_ML_SAMPLES - len(ml_df)
            checks.append((_CAUTION, "ML-Kalibrierung",
                            f"Nur {len(ml_df)} kalibrierte Predictions (≥{MIN_ML_SAMPLES} benötigt)",
                            f"Noch {n_missing} weitere Predictions abwarten (kalibrieren nach 6h)"))
    except Exception:
        warnings.append("predictions-Tabelle nicht auswertbar")

    # ── 9. Log-Stabilität ────────────────────────────────────────────────────
    log_path = ROOT / "logs" / "trading_bot.log"
    log_stats = _parse_logs_for_errors(log_path, days=min(days, 7))

    if log_stats["total_lines"] > 0:
        error_rate = log_stats["error_lines"] / log_stats["total_lines"]
        crashes    = log_stats["traceback_count"]
        criticals  = log_stats["critical_lines"]

        detail = (f"ERROR-Rate={error_rate*100:.2f}% ({log_stats['error_lines']} von "
                  f"{log_stats['total_lines']} Zeilen) | "
                  f"Tracebacks={crashes} | CRITICAL={criticals}")

        if error_rate <= MAX_ERROR_RATE and crashes <= MAX_CRASH_ERRORS and criticals == 0:
            checks.append((_GO, "Log-Stabilität", detail, None))
        elif error_rate <= MAX_ERROR_RATE * 3 and crashes <= MAX_CRASH_ERRORS * 2:
            fix_parts = []
            if crashes > 0:
                fix_parts.append(f"{crashes} Traceback(s) in den letzten {min(days,7)}d untersuchen")
            if error_rate > MAX_ERROR_RATE:
                fix_parts.append("ERROR-Rate leicht erhöht")
            checks.append((_CAUTION, "Log-Stabilität", detail, " | ".join(fix_parts)))
        else:
            fix_parts = []
            if crashes > MAX_CRASH_ERRORS:
                fix_parts.append(f"{crashes} Crashes – Bot crasht regelmäßig")
            if error_rate > MAX_ERROR_RATE * 3:
                fix_parts.append(f"ERROR-Rate {error_rate*100:.1f}% zu hoch")
            if criticals > 0:
                fix_parts.append(f"{criticals} CRITICAL-Fehler")
            checks.append((_NOGO, "Log-Stabilität", detail, " | ".join(fix_parts)))

        if log_stats["crash_markers"]:
            warnings.append("Letzte Traceback-Snippets:")
            for m in log_stats["crash_markers"][:3]:
                warnings.append(f"  → {m[:100]}")
    else:
        warnings.append(f"Log-File {log_path} nicht auswertbar (leer oder nicht vorhanden)")

    con.close()

    # ── 10. Ausgabe ───────────────────────────────────────────────────────────
    print(f"  {'Check':<20} {'Status':<12} Details")
    print("  " + "-" * 78)
    go_count     = 0
    caution_count = 0
    nogo_count   = 0
    action_items = []

    for status, label, detail, fix in checks:
        print(f"{status} {label:<20} {detail}")
        if status == _GO:
            go_count += 1
        elif status == _CAUTION:
            caution_count += 1
            if fix:
                action_items.append(f"  ⚠  {label}: {fix}")
        else:
            nogo_count += 1
            if fix:
                action_items.append(f"  ✗  {label}: {fix}")

    if warnings:
        print()
        print("  Hinweise:")
        for w in warnings:
            print(f"  ⓘ  {w}")

    print()
    _divider("GESAMTURTEIL")
    total_checks = len(checks)
    if nogo_count == 0 and caution_count <= 1:
        print(f"  ██ GO – Bot ist bereit für Live-Betrieb ██")
        print(f"  {go_count}/{total_checks} Checks bestanden, {caution_count} Hinweise")
    elif nogo_count == 0:
        print(f"  ▓▓ FAST BEREIT – Kleine Punkte beheben, dann Live ▓▓")
        print(f"  {go_count}/{total_checks} Checks bestanden, {caution_count} Hinweise, 0 Blocker")
    elif nogo_count <= 1 and caution_count <= 2:
        print(f"  ▒▒ MAINTENANCE NÖTIG – {nogo_count} Blocker beheben ▒▒")
        print(f"  {go_count}/{total_checks} Checks bestanden, {caution_count} Hinweise, {nogo_count} Blocker")
    else:
        print(f"  ░░ NICHT BEREIT – Weiter Paper-traden ░░")
        print(f"  {go_count}/{total_checks} Checks bestanden, {nogo_count} Blocker")

    if action_items:
        print()
        print("  Maßnahmen:")
        for item in action_items:
            print(item)

    _divider()


# ── --run-sweep ───────────────────────────────────────────────────────────────

def cmd_run_sweep(symbol: str):
    _divider(f"Grid-Backtest | {symbol}")
    sys.path.insert(0, str(ROOT))

    try:
        from data_fetcher import fetch_ohlcv
        df = fetch_ohlcv(symbol, "1h", 720)
    except Exception as e:
        print(f"[FEHLER] Daten für {symbol} nicht verfügbar: {e}")
        return

    try:
        from strategies.grid import GridStrategy
        from backtest.engine import run_backtest
        strategy = GridStrategy([{"symbol": symbol, "investment": 200.0, "levels": 8}])
        results = run_backtest(strategy, df, symbol, initial_balance=200.0)
        results.pop("equity_curve", None)
        results.pop("pnls", None)
        print()
        for k, v in results.items():
            print(f"  {k:<25} {v}")
    except Exception as e:
        print(f"[FEHLER] Backtest fehlgeschlagen: {e}")

    _divider()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Grid Trading Bot – Optimizer CLI")
    parser.add_argument("--analyze-trades",    action="store_true")
    parser.add_argument("--calibration-report", action="store_true")
    parser.add_argument("--suggest-params",    action="store_true")
    parser.add_argument("--pattern-mine",      action="store_true")
    parser.add_argument("--run-sweep",         action="store_true")
    parser.add_argument("--ready-for-live",    action="store_true",
                        help="Live-Bereitschaftscheck: Trade-Performance, ML-Kalibrierung, Log-Stabilität")
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
    elif args.ready_for_live:
        cmd_ready_for_live(args.days)
    elif args.run_sweep:
        if not args.symbol:
            print("[FEHLER] --run-sweep benötigt --symbol, z.B. --symbol SOL/USD")
            sys.exit(1)
        cmd_run_sweep(args.symbol)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
