#!/usr/bin/env python3
"""
Nightly analysis routine (runs daily at 05:00 local time via launchd,
~/Library/LaunchAgents/com.tradingbot.nightlytune.plist — requires the Mac
to be awake; needs local trades.db + .env for analysis/Telegram, so a cloud
routine can't run this step).

Workflow
--------
1.  Run full trade analysis (last 7 days) + pattern mining.
2.  Run OOS parameter sweep for each active symbol (180-day window).
3.  Open a GitHub issue with the full findings report.
4.  Send Telegram notification.

Sandboxing guarantee: this script NEVER modifies any file or branch in the
repository.  It is purely observational — reads DB, writes one GitHub issue.
The user decides whether to apply sweep-winner params manually.
"""

import json
import logging
import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [nightly_tune] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(ROOT / "logs" / "nightly_tune.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("nightly_tune")

TODAY = date.today().isoformat()
REPO  = "lucauibk/trading_bot"

# sweep.py aggregates the min-trades gate over the trades of a single invocation.
# 60 was verified against the live 120-day fill rate in #201 (a single all-symbol
# run reached only ~97 aggregate trades for the best config, so the default 100
# gated everything out). Recommendation-only + OOS-Calmar-gated + manually applied,
# so this is a robustness threshold, not a live-risk parameter.
SWEEP_MIN_TRADES = 60
SWEEP_TIMEOUT_SEC = 3600  # one all-symbol run (was 900s per coin, serial)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _read_capture(cmd: list[str], timeout: int = 600) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() or f"[no output — stderr: {r.stderr.strip()[:500]}]"
    except subprocess.TimeoutExpired:
        return "[Timeout]"
    except Exception as e:
        return f"[Error: {e}]"


def _active_symbols() -> List[str]:
    try:
        import yaml
        with open(ROOT / "config" / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("symbols", ["SOL/USD"])
    except Exception:
        return ["SOL/USD"]


def _current_winner_params() -> dict:
    p = ROOT / "config" / "grid_params.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def _open_issue_exists() -> bool:
    """True wenn schon ein offenes Auto-Tune-Issue für heute existiert."""
    result = subprocess.run(
        ["gh", "issue", "list", "--repo", REPO, "--state", "open",
         "--search", f"[Auto-Tune] Nightly findings {TODAY}",
         "--json", "number"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        issues = json.loads(result.stdout or "[]")
        return len(issues) > 0
    return False


# ── Step 1: Analysis ────────────────────────────────────────────────────────────

def run_analysis() -> str:
    log.info("Running trade analysis (last 7 days)…")
    analysis = _read_capture(["python3", "scripts/optimize.py", "--analyze-trades", "--days", "7"])
    log.info("Running pattern mining…")
    patterns = _read_capture(["python3", "scripts/optimize.py", "--pattern-mine", "--days", "30"])
    return (
        "## Trade Analysis (last 7 days)\n\n"
        f"```\n{analysis}\n```\n\n"
        "## Pattern Mining\n\n"
        f"```\n{patterns}\n```\n"
    )


# ── Step 2: Parameter sweep ─────────────────────────────────────────────────────

def run_sweep(symbols: List[str]) -> Tuple[Optional[Dict], str]:
    """
    Run ONE OOS sweep across all active symbols (real cross-symbol aggregation).

    This used to loop one ``--symbol`` invocation per coin, but sweep.py evaluates
    its ``min_trades`` gate on the trade count *aggregated across the symbols of a
    single invocation*. A per-coin call only ever sees that coin's ~9-27 trades, so
    no config could clear the (default 100) gate — the nightly report was
    structurally always empty (#201). On top of that the old code tried to read the
    winning Calmar out of ``r.stdout``, but sweep.py logs to stderr, so the parse
    never matched and best_params stayed None even when a winner.json existed.

    We now do a single invocation over the full active set and read winner.json
    directly (sweep.py already picks the single best cross-symbol config).

    Returns (winner_params_dict | None, report_text).
    Does NOT write or commit anything — winner is reported as a recommendation only.
    """
    parts = ["## Parameter Sweep (OOS — recommendation only, no automatic apply)\n"]
    if not symbols:
        parts.append("No active symbols — sweep skipped.\n")
        return None, "\n".join(parts)

    cmd = ["python3", "scripts/sweep.py",
           "--days", "180", "--train-days", "120", "--jobs", "4",
           "--min-trades", str(SWEEP_MIN_TRADES)]
    for sym in symbols:
        cmd += ["--symbol", sym]

    log.info("Sweep: %s (single cross-symbol run)…", ", ".join(symbols))
    best_params: Optional[dict] = None
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=SWEEP_TIMEOUT_SEC)
        tail = ((r.stdout or "")[-2000:] + (r.stderr or "")[-2000:]).strip()
        parts.append(f"### {', '.join(symbols)}\n\n```\n{tail}\n```\n")

        results_dirs = sorted((ROOT / "results").glob("sweep_*"))
        if results_dirs:
            winner_file = results_dirs[-1] / "winner.json"
            if winner_file.exists():
                best_params = json.loads(winner_file.read_text())
                log.info("Sweep winner: %s", best_params)
    except subprocess.TimeoutExpired:
        parts.append(f"Timeout (>{SWEEP_TIMEOUT_SEC // 60} min) — sweep skipped.\n")
    except Exception as e:
        parts.append(f"Sweep failed: {e}\n")

    return best_params, "\n".join(parts)


# ── Step 3: GitHub issue ────────────────────────────────────────────────────────

def create_issue(analysis_report: str, sweep_report: str,
                 new_params: Optional[Dict]) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    if new_params:
        old_params = _current_winner_params()
        if new_params == old_params:
            params_section = (
                "\n\n## Sweep Result\n\n"
                "Sweep winner found but params are **already identical to current `config/grid_params.json`**. "
                "No action needed.\n"
            )
        else:
            params_section = (
                "\n\n## Sweep Winner — Manual Apply Required\n\n"
                "A better OOS config was found. To apply, copy the block below into "
                "`config/grid_params.json` and restart the bot after reviewing:\n\n"
                f"```json\n{json.dumps(new_params, indent=2)}\n```\n"
            )
    else:
        params_section = (
            "\n\n## Sweep Result\n\n"
            "No sweep winner passed the OOS Calmar gate. Current params remain optimal.\n"
        )

    issue_body = (
        f"# Nightly Analysis Report — {ts}\n\n"
        f"Generated by `scripts/nightly_tune.py` (read-only — no code or config was modified).\n\n"
        + analysis_report
        + sweep_report
        + params_section
        + "\n\n---\n*Auto-generated — no automatic changes were made to any branch or file.*"
    )

    auth_check = subprocess.run(
        ["gh", "auth", "status"], capture_output=True, text=True, check=False
    )
    if auth_check.returncode != 0:
        log.error("gh CLI not authenticated – issue skipped.")
        return ""

    log.info("Creating GitHub issue…")
    result = subprocess.run(
        ["gh", "issue", "create",
         "--repo", REPO,
         "--title", f"[Auto-Tune] Nightly findings {TODAY}",
         "--body", issue_body,
         "--label", "auto-tune"],
        capture_output=True, text=True
    )
    issue_url = result.stdout.strip()
    if issue_url:
        log.info("Issue created: %s", issue_url)
    else:
        log.warning("Issue creation returned no URL. stderr: %s", result.stderr.strip())
    return issue_url


# ── Step 4: Telegram notification ──────────────────────────────────────────────

def notify(issue_url: str, has_sweep_winner: bool):
    try:
        import notifier
        msg_parts = [f"🤖 <b>Nightly Analysis ({TODAY})</b>"]
        if issue_url:
            msg_parts.append(f"📋 Bericht: {issue_url}")
        if has_sweep_winner:
            msg_parts.append("⚙️ Verbesserte Grid-Params gefunden — im Issue beschrieben (manuell anwenden).")
        else:
            msg_parts.append("✅ Keine Parameteränderungen empfohlen.")
        notifier._send("\n".join(msg_parts))
    except Exception as e:
        log.warning("Telegram notify failed: %s", e)


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    log.info("=== Nightly Analysis starting (%s) ===", TODAY)

    if _open_issue_exists():
        log.info("Issue for today already exists — skipping duplicate run.")
        sys.exit(0)

    symbols = _active_symbols()
    log.info("Active symbols: %s", symbols)

    analysis_report = run_analysis()
    new_params, sweep_report = run_sweep(symbols)

    issue_url = create_issue(analysis_report, sweep_report, new_params)
    notify(issue_url, bool(new_params))

    log.info("=== Nightly Analysis complete ===")
    if issue_url:
        print(f"\nIssue: {issue_url}")


if __name__ == "__main__":
    main()
