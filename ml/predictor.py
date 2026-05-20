import logging
import time
from typing import Callable, Dict, List, Optional

import ta as ta_lib

from .data_store import MLDataStore
from .features import extract_features
from .model import LABEL_TO_STR, TradingModel
from .trainer import ModelTrainer, bootstrap_from_history

logger = logging.getLogger("ml.predictor")

MIN_CONFIDENCE  = 0.52  # Mindest-Konfidenz um ML-Vorhersage zu verwenden
RULE_THRESHOLD  = 3     # Score-Schwelle für regelbasiertes Fallback


class MLPredictor:
    """
    Hauptschnittstelle für KI-Vorhersagen.
    Initialisierung einmalig in run(), danach predict(symbol) aufrufen.
    """

    def __init__(self, fetch_ohlcv_fn: Callable):
        self._fetch_ohlcv = fetch_ohlcv_fn
        self._store       = MLDataStore()
        self._models:  Dict[str, TradingModel]  = {}
        self._trainer: Optional[ModelTrainer]   = None

    def initialize(self, symbols: List[str]):
        """Bootstrap-Training beim Start. Lädt vorhandene Modelle, trainiert neue."""
        for sym in symbols:
            self._models[sym] = TradingModel(sym)
        self._trainer = ModelTrainer(self._store, self._models)

        for sym in symbols:
            model = self._models[sym]
            if model.is_ready():
                logger.info("ML-Modell bereits vorhanden für %s (%d Samples)", sym, model._n_samples)
                continue
            logger.info("Bootstrap ML-Modell für %s (1000 Candles)…", sym)
            try:
                df = self._fetch_ohlcv(sym, "1h", 1000)
                bootstrap_from_history(sym, df, self._store, model)
            except Exception as e:
                logger.warning("Bootstrap fehlgeschlagen %s: %s", sym, e)

    def predict(self, symbol: str) -> str:
        """
        Gibt 'up', 'down' oder 'neutral' zurück.
        Kompatibel mit dem bestehenden predict_direction()-Interface.
        """
        try:
            df    = self._fetch_ohlcv(symbol, "1h", 120)
            feats = extract_features(df)
            price = float(df["close"].iloc[-1])
            model = self._models.get(symbol)

            if model and model.is_ready():
                label_int, confidence = model.predict(feats)
                ts = int(time.time())
                if self._trainer:
                    self._trainer.record(symbol, ts, feats, price, label_int)
                    self._trainer.label_and_maybe_retrain(symbol, df)

                if confidence >= MIN_CONFIDENCE:
                    direction = {"sell": "down", "hold": "neutral", "buy": "up"}[LABEL_TO_STR[label_int]]
                    logger.info(
                        "ML %-12s → %-7s (konfidenz=%.2f, n=%d)",
                        symbol, direction.upper(), confidence, model._n_samples,
                    )
                    return direction

                logger.info(
                    "ML %s: Konfidenz %.2f < %.2f → regelbasiertes Fallback",
                    symbol, confidence, MIN_CONFIDENCE,
                )

            result = self._rule_based(df)
            logger.info("Fallback %-12s → %s", symbol, result.upper())
            return result

        except Exception as e:
            logger.warning("ML Fehler %s: %s", symbol, e)
            return "neutral"

    # ── Regelbasiertes Fallback (identisch zu original predict_direction) ──────

    def _rule_based(self, df) -> str:
        try:
            close = df["close"]; high = df["high"]; low = df["low"]
            open_ = df["open"];  volume = df["volume"]

            ema9  = ta_lib.trend.ema_indicator(close, window=9).iloc[-1]
            ema21 = ta_lib.trend.ema_indicator(close, window=21).iloc[-1]
            rsi   = ta_lib.momentum.rsi(close, window=14).iloc[-1]
            mom   = (close.iloc[-1] - close.iloc[-4]) / close.iloc[-4] * 100

            macd_line   = ta_lib.trend.macd(close).iloc[-1]
            macd_signal = ta_lib.trend.macd_signal(close).iloc[-1]
            bb_high     = ta_lib.volatility.bollinger_hband(close).iloc[-1]
            bb_low      = ta_lib.volatility.bollinger_lband(close).iloc[-1]
            price       = close.iloc[-1]
            bb_pct      = (price - bb_low) / (bb_high - bb_low) if bb_high != bb_low else 0.5
            vol_mean    = volume.rolling(20).mean().iloc[-1]
            vol_surge   = volume.iloc[-1] > vol_mean * 1.5

            o = open_.iloc[-1]; h = high.iloc[-1]; l = low.iloc[-1]; c = close.iloc[-1]
            po = open_.iloc[-2]; pc = close.iloc[-2]
            body         = abs(c - o)
            total_range  = (h - l) if h != l else 1e-9
            lower_shadow = min(o, c) - l
            upper_shadow = h - max(o, c)

            score = 0
            if ema9 > ema21:   score += 1
            if ema9 < ema21:   score -= 1
            if mom > 0.5:      score += 1
            if mom < -0.5:     score -= 1
            if rsi < 35:       score += 1
            if rsi > 65:       score -= 1
            if macd_line > macd_signal:  score += 1
            if macd_line < macd_signal:  score -= 1
            if bb_pct < 0.2:   score += 1
            if bb_pct > 0.8:   score -= 1
            if vol_surge and mom > 0: score += 1
            if vol_surge and mom < 0: score -= 1
            if lower_shadow > 2 * body and upper_shadow < body and body / total_range < 0.4: score += 1
            if upper_shadow > 2 * body and lower_shadow < body and body / total_range < 0.4: score -= 1
            if pc < po and c > o and c >= po and o <= pc: score += 1
            if pc > po and c < o and c <= po and o >= pc: score -= 1

            return "up" if score >= RULE_THRESHOLD else "down" if score <= -RULE_THRESHOLD else "neutral"
        except Exception:
            return "neutral"
