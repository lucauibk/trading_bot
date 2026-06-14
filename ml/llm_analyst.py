"""
LLM-basierter Marktanalyst (Claude Haiku).
Wird einmal pro Stunde pro Coin aufgerufen und liefert eine Richtungseinschätzung,
die mit dem LightGBM-Score geblended wird.
"""

import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ml.llm_analyst")

LLM_CONFIDENCE_MIN = 0.60   # Nur verwenden wenn Haiku-Konfidenz ≥ 60%
LLM_WEIGHT         = 0.45   # 45% LLM, 55% LightGBM im Blend
CACHE_SECONDS      = 3600   # Ergebnis 1h cachen (pro Coin-Candle)
MODEL              = "claude-haiku-4-5-20251001"

_cache: dict = {}   # RAM-Cache (schnell); wird beim Start aus DB befüllt

_CACHE_DB = Path(__file__).parents[1] / "data" / "llm_cache.db"


def _db_conn():
    _CACHE_DB.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(_CACHE_DB)
    con.execute("""CREATE TABLE IF NOT EXISTS llm_cache (
        symbol TEXT PRIMARY KEY,
        ts     REAL,
        direction TEXT,
        confidence REAL,
        score  REAL,
        reason TEXT
    )""")
    con.commit()
    return con


def _load_cache_from_db():
    """Füllt den RAM-Cache beim Start aus der DB – verhindert API-Calls nach Neustart."""
    try:
        con = _db_conn()
        rows = con.execute("SELECT symbol, ts, direction, confidence, score, reason FROM llm_cache").fetchall()
        con.close()
        for sym, ts, direction, confidence, score, reason in rows:
            if time.time() - ts < CACHE_SECONDS:
                _cache[sym] = {"ts": ts, "direction": direction,
                               "confidence": confidence, "score": score, "reason": reason}
    except Exception as e:
        logger.debug("LLM Cache-Load fehlgeschlagen: %s", e)


def _save_to_db(symbol: str, result: dict):
    try:
        con = _db_conn()
        con.execute("""INSERT INTO llm_cache (symbol, ts, direction, confidence, score, reason)
                       VALUES (?,?,?,?,?,?)
                       ON CONFLICT(symbol) DO UPDATE SET
                           ts=excluded.ts, direction=excluded.direction,
                           confidence=excluded.confidence, score=excluded.score,
                           reason=excluded.reason""",
                    (symbol, result["ts"], result["direction"],
                     result["confidence"], result["score"], result.get("reason", "")))
        con.commit()
        con.close()
    except Exception as e:
        logger.debug("LLM Cache-Save fehlgeschlagen: %s", e)


_load_cache_from_db()


def _build_prompt(symbol: str, indicators: dict) -> str:
    rsi        = indicators.get("rsi", 50)
    ema9       = indicators.get("ema9", 0)
    ema21      = indicators.get("ema21", 0)
    atr_pct    = indicators.get("atr_pct", 2.0)
    bb_pos     = indicators.get("bb_position", 0.5)
    regime     = indicators.get("regime", "ranging")
    price      = indicators.get("price", 0)
    candles    = indicators.get("last_candles", [])

    candle_str = ""
    for c in candles[-5:]:
        pct = ((c["close"] - c["open"]) / c["open"] * 100) if c.get("open") else 0
        candle_str += f"  {c.get('time','')}: O={c.get('open',0):.4f} H={c.get('high',0):.4f} L={c.get('low',0):.4f} C={c.get('close',0):.4f} ({pct:+.2f}%)\n"

    ema_trend = "bullish" if ema9 > ema21 else "bearish"

    return f"""You are a concise crypto trading signal generator.

Asset: {symbol}
Current price: {price:.4f} USD
Regime: {regime}

Technical indicators:
- RSI(14): {rsi:.1f}  (oversold <35, overbought >65)
- EMA9 vs EMA21: {ema_trend} ({ema9:.4f} vs {ema21:.4f})
- ATR%: {atr_pct:.2f}%
- Bollinger Band position: {bb_pos:.2f}  (0=lower band, 1=upper band)

Last 5 hourly candles:
{candle_str}
Based on these indicators, provide a SHORT-TERM (next 1-4 hours) directional signal.

Respond ONLY with valid JSON, no explanation outside the JSON:
{{"direction": "up"|"neutral"|"down", "confidence": 0.0-1.0, "reason": "max 15 words"}}"""


def analyse(symbol: str, indicators: dict) -> Optional[dict]:
    """
    Ruft Claude Haiku auf und gibt {"direction", "confidence", "score", "reason"} zurück.
    score ist -1.0 (down) … +1.0 (up), skaliert mit confidence.
    Gibt None zurück wenn API-Key fehlt oder Fehler auftritt.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None

    # Cache prüfen
    cached = _cache.get(symbol)
    if cached and (time.time() - cached["ts"]) < CACHE_SECONDS:
        logger.debug("LLM Cache-Hit %s: %s (%.2f)", symbol, cached["direction"], cached["confidence"])
        return cached

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = _build_prompt(symbol, indicators)

        msg = client.messages.create(
            model=MODEL,
            max_tokens=120,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        # Markdown-Codeblocks entfernen falls vorhanden
        if "```" in raw:
            raw = raw.split("```")[-2] if raw.count("```") >= 2 else raw
            raw = raw.lstrip("json").strip()
        data = json.loads(raw)

        direction  = data.get("direction", "neutral")
        confidence = float(data.get("confidence", 0.5))
        reason     = data.get("reason", "")

        score_map = {"up": 1.0, "neutral": 0.0, "down": -1.0}
        score = score_map.get(direction, 0.0) * confidence

        result = {
            "ts": time.time(),
            "direction": direction,
            "confidence": confidence,
            "score": score,
            "reason": reason,
        }
        _cache[symbol] = result
        _save_to_db(symbol, result)

        logger.info(
            "LLM %s → %s (conf=%.2f, score=%+.2f) | %s",
            symbol, direction.upper(), confidence, score, reason,
        )
        return result

    except Exception as e:
        err = str(e).lower()
        if any(k in err for k in ("credit", "billing", "quota", "insufficient", "payment", "permission")):
            logger.error("LLM CREDITS AUFGEBRAUCHT oder API-Fehler: %s", e)
            try:
                import notifier
                notifier._send(
                    "⚠️ <b>Anthropic API – Credits aufgebraucht!</b>\n"
                    "LLM-Analyse deaktiviert. Bot läuft nur noch mit LightGBM weiter.\n"
                    f"Fehler: {e}"
                )
            except Exception:
                pass
        else:
            logger.warning("LLM Fehler %s: %s", symbol, e)
        return None


def blend_scores(lgbm_score: float, lgbm_confidence: float,
                 llm_result: Optional[dict]) -> tuple[float, float]:
    """
    Kombiniert LightGBM- und LLM-Score.
    Gibt (blended_score, blended_confidence) zurück.
    score: -1.0 (down) … +1.0 (up)
    """
    if llm_result is None or llm_result["confidence"] < LLM_CONFIDENCE_MIN:
        return lgbm_score, lgbm_confidence

    llm_score = llm_result["score"]
    blended   = (1 - LLM_WEIGHT) * lgbm_score + LLM_WEIGHT * llm_score
    blended_conf = (1 - LLM_WEIGHT) * lgbm_confidence + LLM_WEIGHT * llm_result["confidence"]

    logger.debug(
        "Blend: LGBM=%+.2f (%.2f) + LLM=%+.2f (%.2f) → %+.2f (%.2f)",
        lgbm_score, lgbm_confidence, llm_score, llm_result["confidence"],
        blended, blended_conf,
    )
    return blended, blended_conf
