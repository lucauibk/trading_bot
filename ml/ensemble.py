"""
Ensemble blending: LightGBM calibrated probability + Claude Haiku LLM signal.

Key improvements over the old linear blend:
1. Disagreement handling: LGBM=BUY + LLM=SELL → neutral (no trade)
2. Bayesian combination: LGBM calibrated proba as prior, LLM as likelihood multiplier
3. LLM confidence used ONCE (as gating threshold, not multiplying the score again)
4. Anthropic prompt caching to reduce token cost by ~90%
"""

import json
import logging
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("ml.ensemble")

# ── Config ────────────────────────────────────────────────────────────────────
LLM_MIN_CONF = 0.65          # LLM only used if its confidence ≥ this
LLM_LIKELIHOOD_STRENGTH = 2.0  # How much LLM shifts the posterior (multiplier strength)
DISAGREEMENT_NEUTRAL = True  # LGBM↑ + LLM↓ → neutral
MIN_CONFIDENCE = 0.45         # Minimum blended confidence to trade
CACHE_TTL = 3600              # LLM cache TTL in seconds
MODEL = "claude-haiku-4-5-20251001"

_CACHE_DB = Path("data/llm_cache.db")
_cache: dict = {}  # RAM cache


# ── DB Cache ──────────────────────────────────────────────────────────────────

def _db_conn() -> sqlite3.Connection:
    _CACHE_DB.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(_CACHE_DB)
    con.execute("""
        CREATE TABLE IF NOT EXISTS llm_cache (
            symbol     TEXT PRIMARY KEY,
            ts         REAL,
            direction  TEXT,
            confidence REAL,
            score      REAL,
            reason     TEXT
        )
    """)
    con.commit()
    return con


def _load_cache():
    global _cache
    try:
        con = _db_conn()
        rows = con.execute("SELECT symbol, ts, direction, confidence, score, reason FROM llm_cache").fetchall()
        con.close()
        for sym, ts, direction, confidence, score, reason in rows:
            if time.time() - ts < CACHE_TTL:
                _cache[sym] = {"ts": ts, "direction": direction,
                               "confidence": confidence, "score": score, "reason": reason}
    except Exception as e:
        logger.debug("LLM cache load failed: %s", e)


def _save_cache(symbol: str, result: dict):
    try:
        con = _db_conn()
        con.execute("""
            INSERT INTO llm_cache (symbol, ts, direction, confidence, score, reason)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(symbol) DO UPDATE SET
                ts=excluded.ts, direction=excluded.direction,
                confidence=excluded.confidence, score=excluded.score,
                reason=excluded.reason
        """, (symbol, result["ts"], result["direction"],
              result["confidence"], result["score"], result.get("reason", "")))
        con.commit()
        con.close()
    except Exception as e:
        logger.debug("LLM cache save failed: %s", e)


_load_cache()


# ── Claude Haiku call with Prompt Caching ────────────────────────────────────

_SYSTEM_PROMPT = """You are a concise crypto trading signal generator.
Output ONLY valid JSON with keys: direction ("up"/"down"/"neutral"), confidence (0.0-1.0), reason (≤15 words).
Base your signal on the provided technical data. Be decisive – avoid "neutral" unless genuinely unclear."""

_FEW_SHOTS = """Examples:
{"direction":"up","confidence":0.72,"reason":"RSI oversold, bullish engulfing near BB lower band"}
{"direction":"down","confidence":0.68,"reason":"RSI overbought, MACD bearish cross, high funding rate"}
{"direction":"neutral","confidence":0.51,"reason":"Mixed signals, ranging regime, low volume"}"""


def _call_llm(symbol: str, indicators: dict) -> Optional[dict]:
    """Call Claude Haiku with prompt caching. Returns parsed result or None."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        price = indicators.get("price", 0)
        rsi = indicators.get("rsi", 50)
        ema9 = indicators.get("ema9", price)
        ema21 = indicators.get("ema21", price)
        atr_pct = indicators.get("atr_pct", 2.0)
        bb_pos = indicators.get("bb_position", 0.5)
        regime = indicators.get("regime", "ranging")
        funding = indicators.get("funding_rate", 0)
        funding_z = indicators.get("funding_z", 0)
        btc_r1h = indicators.get("btc_return_1h", 0)
        candles = indicators.get("last_candles", [])

        candle_lines = ""
        for c in candles[-5:]:
            pct = ((c["close"] - c["open"]) / c["open"] * 100) if c.get("open") else 0
            candle_lines += f"  {c.get('time','')}: {c.get('close',0):.4f} ({pct:+.2f}%)\n"

        user_content = f"""Asset: {symbol} @ {price:.4f} USD | Regime: {regime}
RSI={rsi:.0f} | EMA9/21={"above" if ema9 > ema21 else "below"} | BB={bb_pos:.2f} | ATR={atr_pct:.2f}%
Funding={funding*100:.4f}% (z={funding_z:.2f}) | BTC 1h={btc_r1h*100:+.2f}%
Last 5 candles:
{candle_lines}
What is the short-term direction?"""

        response = client.messages.create(
            model=MODEL,
            max_tokens=80,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT + "\n\n" + _FEW_SHOTS,
                    "cache_control": {"type": "ephemeral"},  # Anthropic prompt caching
                }
            ],
            messages=[{"role": "user", "content": user_content}],
        )
        raw = response.content[0].text.strip()

        # Robust JSON extraction
        match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
        if not match:
            logger.warning("LLM non-JSON response for %s: %s", symbol, raw[:100])
            return None

        data = json.loads(match.group())
        direction = data.get("direction", "neutral").lower()
        confidence = float(data.get("confidence", 0.5))
        reason = str(data.get("reason", ""))

        if direction not in ("up", "down", "neutral"):
            direction = "neutral"

        score = confidence if direction == "up" else (-confidence if direction == "down" else 0.0)

        result = {
            "direction": direction,
            "confidence": confidence,
            "score": score,
            "reason": reason,
            "ts": time.time(),
        }
        logger.info("[LLM] %s → %s (conf=%.2f) | %s", symbol, direction.upper(), confidence, reason)
        return result

    except Exception as e:
        logger.warning("LLM call failed for %s: %s", symbol, e)
        return None


def get_llm_signal(symbol: str, indicators: dict, force_refresh: bool = False) -> Optional[dict]:
    """Return cached or fresh LLM signal for a symbol."""
    cached = _cache.get(symbol)
    if not force_refresh and cached and time.time() - cached.get("ts", 0) < CACHE_TTL:
        return cached

    result = _call_llm(symbol, indicators)
    if result:
        _cache[symbol] = result
        _save_cache(symbol, result)
    return result


# ── Bayesian Blending ─────────────────────────────────────────────────────────

def blend(
    lgbm_proba: np.ndarray,     # shape (3,): [P(sell), P(hold), P(buy)]
    llm_signal: Optional[dict],  # from get_llm_signal()
    label_buy: int = 2,
    label_sell: int = 0,
    label_hold: int = 1,
) -> Tuple[str, float, float]:
    """
    Bayesian combination of LGBM calibrated probability and LLM signal.

    LGBM provides calibrated proba as prior.
    LLM acts as likelihood multiplier if conf ≥ LLM_MIN_CONF.
    If LGBM and LLM disagree on direction → return neutral.

    Returns (direction, blended_score, blended_confidence)
    """
    buy_p = float(lgbm_proba[label_buy])
    sell_p = float(lgbm_proba[label_sell])
    hold_p = float(lgbm_proba[label_hold])

    lgbm_dir = "buy" if buy_p > sell_p and buy_p > hold_p else (
        "sell" if sell_p > buy_p and sell_p > hold_p else "hold"
    )
    lgbm_conf = max(buy_p, sell_p, hold_p)

    if llm_signal is None or llm_signal.get("confidence", 0) < LLM_MIN_CONF:
        # No LLM signal – use LGBM only
        score = buy_p - sell_p
        direction = "up" if score > 0.15 else ("down" if score < -0.15 else "neutral")
        return direction, score, lgbm_conf

    llm_dir = llm_signal.get("direction", "neutral")
    llm_conf = float(llm_signal.get("confidence", 0))

    # Disagreement guard
    if DISAGREEMENT_NEUTRAL:
        lgbm_binary = "up" if lgbm_dir == "buy" else ("down" if lgbm_dir == "sell" else "neutral")
        if lgbm_binary != "neutral" and llm_dir != "neutral" and lgbm_binary != llm_dir:
            logger.info("[ENSEMBLE] %s Disagreement: LGBM=%s LLM=%s → neutral",
                        "?", lgbm_binary.upper(), llm_dir.upper())
            return "neutral", 0.0, 0.3

    # Bayesian posterior: P(direction) ∝ prior × likelihood
    # LLM likelihood: if LLM says "up", multiply P(buy) by strength
    posterior = np.array([sell_p, hold_p, buy_p], dtype=float)

    strength = LLM_LIKELIHOOD_STRENGTH
    if llm_dir == "up":
        posterior[label_buy] *= strength
    elif llm_dir == "down":
        posterior[label_sell] *= strength
    # Normalize
    total = posterior.sum()
    if total > 0:
        posterior /= total

    final_buy = posterior[label_buy]
    final_sell = posterior[label_sell]
    score = final_buy - final_sell

    if abs(score) < 0.15:
        direction = "neutral"
    elif score > 0:
        direction = "up"
    else:
        direction = "down"

    blended_conf = max(posterior)
    return direction, float(score), float(blended_conf)
