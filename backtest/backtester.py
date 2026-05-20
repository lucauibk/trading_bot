"""
Backtesting Engine mit Charts und HTML-Report.
Aufruf: python3 -m backtest.backtester --strategy ema --symbol SOL/USDT
"""

import argparse
import base64
import io
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml

from src.data.fetcher import get_ohlcv_since
from src.risk.risk_manager import RiskManager
from src.strategy.base_strategy import Signal


# ── Backtest-Kern ─────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, strategy, risk: RiskManager,
                 initial_capital: float = 1000.0) -> dict:
    capital  = initial_capital
    trades   = []
    equity   = [capital]
    open_pos = None

    df = df.copy()
    df.dropna(inplace=True)

    for i in range(50, len(df)):
        window = df.iloc[:i + 1]
        row    = window.iloc[-1]
        price  = row["close"]
        ts     = window.index[-1]

        # ── Exit ──
        if open_pos:
            exit_reason = None
            if price <= open_pos["stop_loss"]:
                exit_reason = "stop_loss"
            elif price >= open_pos["take_profit"]:
                exit_reason = "take_profit"
            elif hasattr(strategy, "should_exit") and strategy.should_exit(window, Signal.LONG):
                exit_reason = "signal_exit"

            if exit_reason:
                pnl = (price - open_pos["entry"]) * open_pos["qty"]
                fee = (price + open_pos["entry"]) * open_pos["qty"] * 0.001
                pnl -= fee
                capital += pnl
                trades.append({
                    "entry_time": open_pos["entry_time"],
                    "exit_time":  ts,
                    "entry":      open_pos["entry"],
                    "exit":       price,
                    "pnl":        pnl,
                    "reason":     exit_reason,
                })
                open_pos = None

        # ── Entry ──
        if not open_pos:
            sig = strategy.generate_signal(window)
            if sig.is_valid:
                size = risk.calculate_position_size(capital, sig.entry, sig.stop_loss)
                if size.valid and risk.check_daily_drawdown(capital):
                    open_pos = {
                        "entry_time":  ts,
                        "entry":       sig.entry,
                        "stop_loss":   sig.stop_loss,
                        "take_profit": sig.take_profit,
                        "qty":         size.quantity,
                    }

        equity.append(capital)

    return _summarize(trades, equity, initial_capital, df.index[50:])


def _summarize(trades: list, equity: list, initial_capital: float, index) -> dict:
    if not trades:
        return {"error": "Keine Trades", "trades": [], "equity": equity, "index": index}

    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    total  = sum(t["pnl"] for t in trades)
    eq     = np.array(equity)
    peak   = np.maximum.accumulate(eq)
    dd_arr = (eq - peak) / peak
    max_dd = float(dd_arr.min()) * 100

    loss_sum = sum(t["pnl"] for t in losses)
    pf = abs(sum(t["pnl"] for t in wins) / loss_sum) if loss_sum != 0 else float("inf")

    returns = pd.Series(equity).pct_change().dropna()
    sharpe  = (returns.mean() / returns.std() * np.sqrt(24 * 365)
               if returns.std() > 0 else 0)

    # Bestes / schlechtestes Monat
    trade_df   = pd.DataFrame(trades)
    trade_df["month"] = pd.to_datetime(trade_df["exit_time"]).dt.to_period("M")
    monthly    = trade_df.groupby("month")["pnl"].sum()
    best_month  = f"{monthly.idxmax()} ({monthly.max():+.2f} USDT)" if not monthly.empty else "–"
    worst_month = f"{monthly.idxmin()} ({monthly.min():+.2f} USDT)" if not monthly.empty else "–"

    return {
        "metrics": {
            "Trades":          len(trades),
            "Win-Rate":        f"{len(wins)/len(trades)*100:.1f}%",
            "Profit Factor":   f"{pf:.2f}",
            "Sharpe Ratio":    f"{sharpe:.2f}",
            "Gesamt PnL":      f"{total:+.2f} USDT",
            "Return":          f"{total/initial_capital*100:+.1f}%",
            "Max Drawdown":    f"{max_dd:.1f}%",
            "Avg Win":         f"{np.mean([t['pnl'] for t in wins]):.2f} USDT" if wins else "–",
            "Avg Loss":        f"{np.mean([t['pnl'] for t in losses]):.2f} USDT" if losses else "–",
            "Bester Monat":    best_month,
            "Schlechtester Monat": worst_month,
        },
        "trades":   trades,
        "equity":   equity,
        "dd_arr":   dd_arr.tolist(),
        "index":    index,
        "monthly":  monthly,
    }


# ── Charts ────────────────────────────────────────────────────────────────────

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def build_charts(result: dict, symbol: str, strategy_name: str) -> dict:
    equity  = result["equity"]
    dd_arr  = result["dd_arr"]
    monthly = result["monthly"]
    index   = result["index"]

    idx = list(index)[:len(equity)]

    # ── Equity + Drawdown ──
    fig = plt.figure(figsize=(14, 7))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(idx, equity[:len(idx)], color="#00c896", linewidth=1.5, label="Equity")
    ax1.fill_between(idx, equity[:len(idx)], alpha=0.15, color="#00c896")
    ax1.set_title(f"{symbol} | {strategy_name} – Equity Curve", fontsize=13, pad=10)
    ax1.set_ylabel("Kapital (USDT)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xticklabels([])

    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(idx, [d * 100 for d in dd_arr[:len(idx)]], 0,
                     color="#ff4d4d", alpha=0.6)
    ax2.set_ylabel("Drawdown %")
    ax2.set_xlabel("Zeit")
    ax2.grid(alpha=0.3)
    plt.xticks(rotation=30)

    equity_b64 = _fig_to_b64(fig)
    plt.close(fig)

    # ── Monthly Heatmap ──
    if not monthly.empty:
        monthly_df = monthly.reset_index()
        monthly_df.columns = ["month", "pnl"]
        monthly_df["year"]  = monthly_df["month"].dt.year
        monthly_df["month_n"] = monthly_df["month"].dt.month
        pivot = monthly_df.pivot(index="year", columns="month_n", values="pnl").fillna(0)
        month_names = ["Jan","Feb","Mär","Apr","Mai","Jun",
                       "Jul","Aug","Sep","Okt","Nov","Dez"]
        pivot.columns = [month_names[m-1] for m in pivot.columns]

        fig2, ax = plt.subplots(figsize=(14, max(3, len(pivot) * 0.8)))
        vmax = max(abs(pivot.values.max()), abs(pivot.values.min()), 1)
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                       vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:+.0f}", ha="center", va="center",
                        fontsize=8, color="black")
        ax.set_title("Monthly Returns Heatmap (USDT)", fontsize=12)
        plt.colorbar(im, ax=ax, label="PnL USDT")
        heatmap_b64 = _fig_to_b64(fig2)
        plt.close(fig2)
    else:
        heatmap_b64 = ""

    return {"equity": equity_b64, "heatmap": heatmap_b64}


# ── HTML-Report ───────────────────────────────────────────────────────────────

def save_html_report(result: dict, charts: dict, symbol: str,
                     strategy_name: str, since: str):
    metrics = result["metrics"]
    trades  = result["trades"]

    rows = "".join(
        f"<tr><td>{t['entry_time'].strftime('%Y-%m-%d %H:%M') if hasattr(t['entry_time'],'strftime') else t['entry_time']}</td>"
        f"<td>{t['exit_time'].strftime('%Y-%m-%d %H:%M') if hasattr(t['exit_time'],'strftime') else t['exit_time']}</td>"
        f"<td>{t['entry']:.4f}</td><td>{t['exit']:.4f}</td>"
        f"<td style='color:{'#00c896' if t['pnl']>0 else '#ff4d4d'}'>{t['pnl']:+.2f}</td>"
        f"<td>{t['reason']}</td></tr>"
        for t in trades[-50:]
    )

    metric_rows = "".join(
        f"<tr><td>{k}</td><td><b>{v}</b></td></tr>"
        for k, v in metrics.items()
    )

    heatmap_section = (
        f'<h2>Monthly Returns</h2><img src="data:image/png;base64,{charts["heatmap"]}" style="max-width:100%">'
        if charts["heatmap"] else ""
    )

    html = f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<title>Backtest Report – {symbol}</title>
<style>
  body {{ font-family: Arial, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
  h1, h2 {{ color: #00c896; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th {{ background: #16213e; padding: 8px 12px; text-align: left; color: #00c896; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #2a2a4a; }}
  tr:hover {{ background: #16213e; }}
  img {{ border-radius: 8px; margin: 8px 0; }}
  .badge {{ background: #00c896; color: #000; padding: 2px 8px; border-radius: 4px; font-size: 12px; }}
</style>
</head>
<body>
<h1>📊 Backtest Report</h1>
<p><span class="badge">{symbol}</span> &nbsp;
   <span class="badge">{strategy_name}</span> &nbsp;
   <span class="badge">seit {since[:10]}</span> &nbsp;
   <span class="badge">generiert {datetime.now().strftime('%Y-%m-%d %H:%M')}</span></p>

<h2>Metriken</h2>
<table><tr><th>Kennzahl</th><th>Wert</th></tr>{metric_rows}</table>

<h2>Equity Curve & Drawdown</h2>
<img src="data:image/png;base64,{charts['equity']}" style="max-width:100%">

{heatmap_section}

<h2>Trade-History (letzte 50)</h2>
<table>
  <tr><th>Entry Zeit</th><th>Exit Zeit</th><th>Entry</th><th>Exit</th><th>PnL</th><th>Grund</th></tr>
  {rows}
</table>
</body></html>"""

    out = Path("backtest_report.html")
    out.write_text(html, encoding="utf-8")
    print(f"\n  Report gespeichert: {out.resolve()}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["ema", "rsi"], default="rsi")
    parser.add_argument("--symbol",   default="SOL/USDT")
    parser.add_argument("--since",    default="2024-01-01T00:00:00Z")
    parser.add_argument("--capital",  type=float, default=1000.0)
    args = parser.parse_args()

    with open("config/strategy_params.yaml") as f:
        params = yaml.safe_load(f)
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    if args.strategy == "ema":
        from src.strategy.ema_crossover import EMACrossoverStrategy
        strategy = EMACrossoverStrategy(params["ema_crossover"])
    else:
        from src.strategy.rsi_mean_rev import RSIMeanRevStrategy
        strategy = RSIMeanRevStrategy(params["rsi_mean_rev"])

    risk = RiskManager(cfg.get("risk", {}), args.capital)

    print(f"Backtest | {args.symbol} | {strategy.name()} | seit {args.since[:10]}")
    print("Lade Daten…")
    df = get_ohlcv_since(args.symbol, "1h", args.since)
    print(f"  {len(df)} Candles geladen\n")

    result = run_backtest(df, strategy, risk, args.capital)

    if "error" in result:
        print(f"  {result['error']}")
        sys.exit(0)

    print("=" * 48)
    for k, v in result["metrics"].items():
        print(f"  {k:<22} {v}")
    print("=" * 48)

    print("\nErstelle Charts…")
    charts = build_charts(result, args.symbol, strategy.name())
    save_html_report(result, charts, args.symbol, strategy.name(), args.since)
