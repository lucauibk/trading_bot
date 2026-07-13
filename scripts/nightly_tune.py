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
    Run OOS sweep for each symbol.
    Returns (winner_params_dict | None, report_text).
    Does NOT write or commit anything — winner is reported as a recommendation only.
    """
    best_calmar = -999.0
    best_params: dict | None = None
    parts = ["## Parameter Sweep (OOS — recommendation only, no automatic apply)\n"]

    for sym in symbols:
        log.info("Sweep: %s…", sym)
        try:
            r = subprocess.run(
                ["python3", "scripts/sweep.py",
                 "--symbol", sym,
                 "--days", "180", "--train-days", "120", "--jobs", "4"],
                capture_output=True, text=True, timeout=900,
            )
            # sweep.py loggt via logging → stderr; stdout ist praktisch leer (#128)
            output = (r.stdout + r.stderr)[-3000:]
            parts.append(f"### {sym}\n\n```\n{output}\n```\n")

            results_dirs = sorted((ROOT / "results").glob("sweep_*"))
            if results_dirs:
                winner_file = results_dirs[-1] / "winner.json"
                meta_file = results_dirs[-1] / "winner_meta.json"
                if winner_file.exists() and meta_file.exists():
                    w = json.loads(winner_file.read_text())
                    try:
                        calmar = float(json.loads(meta_file.read_text())["median_calmar"])
                    except (KeyError, ValueError, json.JSONDecodeError):
                        calmar = None
                    if calmar is not None and calmar > best_calmar:
                        best_calmar = calmar
                        best_params = w
                        log.info("New best config from %s: Calmar=%.2f", sym, calmar)
        except subprocess.TimeoutExpired:
            parts.append(f"### {sym}\n\nTimeout (>15 min) — skipped.\n")
        except Exception as e:
            parts.append(f"### {sym}\n\nFailed: {e}\n")

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
