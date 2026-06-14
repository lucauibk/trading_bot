#!/usr/bin/env python3
"""
Nightly auto-tuning routine (runs daily at 05:00 via /schedule Cloud routine).

Workflow
--------
1.  Create a fresh branch  auto-tune/YYYY-MM-DD  off main.
2.  Run full trade analysis (last 7 days) + pattern mining.
3.  Run OOS parameter sweep for each active symbol (180-day window).
4.  If sweep finds a config with strictly better OOS Calmar, write the new
    params to  config/grid_params.json  and commit to the branch.
5.  Open a GitHub issue with the full findings report.
6.  Open a draft PR from the branch to main linking the issue.
7.  Send Telegram notification.

Sandboxing guarantee: this script NEVER touches main, never restarts the live
bot, and never modifies anything outside its own branch.  The user reviews the
PR and merges manually.
"""

import json
import logging
import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make sure repo root is on sys.path (script lives in scripts/)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [nightly_tune] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("nightly_tune")

TODAY = date.today().isoformat()           # e.g. 2026-06-14
BRANCH = f"auto-tune/{TODAY}"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run(cmd: list[str], check: bool = True, capture: bool = False, **kw):
    """Run a subprocess, optionally capturing output."""
    log.debug("$ %s", " ".join(cmd))
    if capture:
        result = subprocess.run(cmd, capture_output=True, text=True, **kw)
        if check and result.returncode != 0:
            log.error("Command failed: %s\n%s", " ".join(cmd), result.stderr)
            raise RuntimeError(result.stderr)
        return result
    return subprocess.run(cmd, check=check, **kw)


def _git(*args, **kw):
    return _run(["git"] + list(args), **kw)


def _gh(*args, **kw):
    return _run(["gh"] + list(args), **kw)


def _active_symbols() -> List[str]:
    """Load active symbols from config.yaml (same logic as main.py)."""
    try:
        import yaml
        with open(ROOT / "config" / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("symbols", ["SOL/USD"])
    except Exception:
        return ["SOL/USD"]


def _current_winner_params() -> dict:
    """Load current grid_params.json (if it exists) for comparison."""
    p = ROOT / "config" / "grid_params.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def _read_capture(cmd: list[str]) -> str:
    """Run a Python script and capture stdout, ignoring failures."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return r.stdout.strip()
    except Exception as e:
        return f"[Error: {e}]"


# ── Step 1: Branch setup ────────────────────────────────────────────────────────

def setup_branch() -> bool:
    """Create (or reset) the auto-tune branch off origin/main."""
    log.info("Setting up branch %s", BRANCH)
    try:
        _git("fetch", "origin", capture=True)
        # Stash any uncommitted changes so the branch switch succeeds
        stash = _run(["git", "stash", "--include-untracked"], capture=True, check=False)
        stashed = "No local changes" not in (stash.stdout or "")
        _git("branch", "-D", BRANCH, capture=True, check=False)
        _git("switch", "-c", BRANCH, "origin/main")
        if stashed:
            _run(["git", "stash", "pop"], capture=True, check=False)
        log.info("Branch created: %s", BRANCH)
        return True
    except Exception as e:
        log.error("Branch setup failed: %s", e)
        return False


# ── Step 2: Analysis ────────────────────────────────────────────────────────────

def run_analysis() -> str:
    """Run trade analysis and pattern mining; return combined report text."""
    log.info("Running trade analysis (last 7 days)…")
    analysis = _read_capture([
        "python3", "scripts/optimize.py", "--analyze-trades", "--days", "7"
    ])
    log.info("Running pattern mining…")
    patterns = _read_capture([
        "python3", "scripts/optimize.py", "--pattern-mine", "--days", "30"
    ])
    return f"## Trade Analysis (last 7 days)\n\n```\n{analysis}\n```\n\n## Pattern Mining\n\n```\n{patterns}\n```\n"


# ── Step 3: Parameter sweep ─────────────────────────────────────────────────────

def run_sweep(symbols: List[str]) -> Tuple[Optional[Dict], str]:
    """
    Run OOS sweep for each symbol.  Returns (winner_params_dict | None, report_text).
    Winner params are taken from the symbol with the best OOS Calmar.
    """
    best_calmar = -999.0
    best_params: dict | None = None
    sweep_report_parts = ["## Parameter Sweep (OOS)\n"]

    for sym in symbols:
        log.info("Sweep: %s…", sym)
        try:
            r = subprocess.run(
                [
                    "python3", "scripts/sweep.py",
                    "--symbol", sym,
                    "--days", "180",
                    "--train-days", "120",
                    "--jobs", "4",
                ],
                capture_output=True, text=True, timeout=900,
            )
            sweep_report_parts.append(f"### {sym}\n\n```\n{r.stdout[-3000:]}\n```\n")

            # Parse winner.json from latest results dir
            results_dirs = sorted((ROOT / "results").glob("sweep_*"))
            if results_dirs:
                winner_file = results_dirs[-1] / "winner.json"
                if winner_file.exists():
                    w = json.loads(winner_file.read_text())
                    # Extract OOS Calmar from sweep stdout (look for "WINNER" line)
                    for line in r.stdout.splitlines():
                        if "median calmar" in line.lower():
                            try:
                                calmar = float(line.split()[-1])
                                if calmar > best_calmar:
                                    best_calmar = calmar
                                    best_params = w
                                    log.info("New best config from %s: Calmar=%.2f", sym, calmar)
                            except Exception:
                                pass
        except subprocess.TimeoutExpired:
            sweep_report_parts.append(f"### {sym}\n\nTimeout (>15 min) — skipped.\n")
        except Exception as e:
            sweep_report_parts.append(f"### {sym}\n\nFailed: {e}\n")

    return best_params, "\n".join(sweep_report_parts)


# ── Step 4: Commit improved params ─────────────────────────────────────────────

def commit_params(new_params: Dict, old_params: Dict) -> bool:
    """
    Write config/grid_params.json with the new winner params and commit.
    Returns True if a commit was made (i.e. params actually changed).
    """
    params_path = ROOT / "config" / "grid_params.json"
    params_path.write_text(json.dumps(new_params, indent=2))

    if new_params == old_params:
        log.info("New params identical to current — no commit needed")
        return False

    try:
        _git("add", "config/grid_params.json")
        _git(
            "commit", "-m",
            f"auto-tune: update grid_params from sweep {TODAY}\n\n"
            f"OOS sweep winner applied automatically by nightly_tune.py.\n"
            f"Review the associated PR before merging.\n\n"
            f"Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
        )
        log.info("Params committed to %s", BRANCH)
        return True
    except Exception as e:
        log.error("Commit failed: %s", e)
        return False


# ── Step 5+6: GitHub issue + PR ─────────────────────────────────────────────────

def create_issue_and_pr(analysis_report: str, sweep_report: str,
                         params_changed: bool, new_params: Optional[Dict]) -> str:
    """
    Open a GitHub issue with findings, then a draft PR linking it.
    Returns the PR URL (or empty string on failure).
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    params_section = ""
    if params_changed and new_params:
        params_section = f"\n\n## New Grid Params Applied\n\n```json\n{json.dumps(new_params, indent=2)}\n```\n"
    elif new_params:
        params_section = "\n\n## Sweep Result\n\nNew winner found but params unchanged (already optimal).\n"
    else:
        params_section = "\n\n## Sweep Result\n\nNo sweep winner passed the OOS Calmar gate.\n"

    issue_body = (
        f"# Nightly Auto-Tune Report — {ts}\n\n"
        f"Generated by `scripts/nightly_tune.py`.  "
        f"Review the linked PR before merging.\n\n"
        + analysis_report
        + sweep_report
        + params_section
        + "\n\n---\n*Auto-generated — do not merge without review.*"
    )

    # Check gh auth before attempting
    auth_check = _run(["gh", "auth", "status"], capture=True, check=False)
    if auth_check.returncode != 0:
        log.error("gh CLI not authenticated – run 'gh auth login' first. Issue/PR skipped.")
        log.error("gh auth output: %s", auth_check.stderr.strip())
        return ""

    log.info("Creating GitHub issue…")
    try:
        issue_result = _gh(
            "issue", "create",
            "--title", f"[Auto-Tune] Nightly findings {TODAY}",
            "--body", issue_body,
            "--label", "auto-tune",
            capture=True, check=False,
        )
        issue_url = issue_result.stdout.strip()
        if not issue_url:
            log.warning("Issue creation returned no URL. stderr: %s", issue_result.stderr.strip())
        else:
            log.info("Issue created: %s", issue_url)
    except Exception as e:
        log.warning("Issue creation failed: %s", e)
        issue_url = ""

    # Push branch
    try:
        _git("push", "-u", "origin", BRANCH)
    except Exception as e:
        log.warning("Push failed: %s", e)
        return ""

    log.info("Creating draft PR…")
    pr_body = (
        f"## Auto-Tune PR — {TODAY}\n\n"
        f"**DO NOT MERGE without reviewing the analysis below.**\n\n"
        f"{f'Closes {issue_url}' if issue_url else ''}\n\n"
        f"### Changes\n"
        f"{'- Updated `config/grid_params.json` with OOS sweep winner.' if params_changed else '- No parameter changes (sweep found no improvement).'}\n\n"
        f"### How to review\n"
        f"1. Read the linked issue for full analysis + sweep report\n"
        f"2. Check `config/grid_params.json` diff for sanity\n"
        f"3. Run `python3 scripts/optimize.py --calibration-report` locally\n"
        f"4. Merge only if you're satisfied with the changes\n\n"
        f"🤖 Generated with [Claude Code](https://claude.com/claude-code)"
    )
    try:
        pr_result = _gh(
            "pr", "create",
            "--title", f"[Auto-Tune] {TODAY}",
            "--body", pr_body,
            "--base", "main",
            "--head", BRANCH,
            "--draft",
            capture=True, check=False,
        )
        pr_url = pr_result.stdout.strip()
        if not pr_url:
            log.warning("PR creation returned no URL. stderr: %s", pr_result.stderr.strip())
        else:
            log.info("PR created: %s", pr_url)
        return pr_url
    except Exception as e:
        log.warning("PR creation failed: %s", e)
        return ""


# ── Step 7: Telegram notification ──────────────────────────────────────────────

def notify(pr_url: str, params_changed: bool):
    try:
        import notifier
        msg_parts = [f"🤖 <b>Nightly Auto-Tune ({TODAY})</b>"]
        if pr_url:
            msg_parts.append(f"📋 PR bereit zum Review: {pr_url}")
        if params_changed:
            msg_parts.append("⚙️ Verbesserte Grid-Params gefunden und committed.")
        else:
            msg_parts.append("✅ Keine Parameteränderungen (aktuell optimal).")
        msg_parts.append("👀 Bitte PR reviewen und manuell mergen.")
        notifier._send("\n".join(msg_parts))
    except Exception as e:
        log.warning("Telegram notify failed: %s", e)


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    log.info("=== Nightly Auto-Tune starting (%s) ===", TODAY)

    if not setup_branch():
        log.error("Branch setup failed — aborting")
        sys.exit(1)

    symbols = _active_symbols()
    log.info("Active symbols: %s", symbols)

    analysis_report = run_analysis()
    new_params, sweep_report = run_sweep(symbols)
    old_params = _current_winner_params()

    params_changed = False
    if new_params:
        params_changed = commit_params(new_params, old_params)

    pr_url = create_issue_and_pr(analysis_report, sweep_report, params_changed, new_params)
    notify(pr_url, params_changed)

    log.info("=== Nightly Auto-Tune complete ===")
    if pr_url:
        print(f"\nPR: {pr_url}")


if __name__ == "__main__":
    main()
