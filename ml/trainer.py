import logging
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .data_store import MLDataStore
from .features import extract_features
from .model import TradingModel

logger = logging.getLogger("ml.trainer")

LOOKFORWARD_H   = 12     # Timeout in Candles (12h) für Triple-Barrier
RETRAIN_EVERY_N = 50     # Retrain nach N neuen gelabelten Samples

# Triple-Barrier-Parameter: Faktoren auf ATR als Schwelle
TB_UPPER_ATR = 2.0   # Gewinn-Barriere = 2× ATR
TB_LOWER_ATR = 1.5   # Verlust-Barriere = 1.5× ATR (asymmetrisch: enger Stop)


def _compute_label_triple_barrier(
    df: pd.DataFrame, idx: int, atr_pct: float
) -> int:
    """
    Triple-Barrier-Methode (López de Prado):
    0=sell (untere Barriere getroffen), 1=hold (Timeout), 2=buy (obere Barriere).
    Verwendet ATR-skalierte Barrieren statt fixer Prozentwerte.
    """
    if idx + LOOKFORWARD_H >= len(df):
        return 1
    p0 = float(df["close"].iloc[idx])
    upper = p0 * (1 + TB_UPPER_ATR * atr_pct)
    lower = p0 * (1 - TB_LOWER_ATR * atr_pct)

    for j in range(idx + 1, min(idx + 1 + LOOKFORWARD_H, len(df))):
        h = float(df["high"].iloc[j])
        l = float(df["low"].iloc[j])
        if h >= upper:
            return 2
        if l <= lower:
            return 0
    return 1


def _get_atr_pct(df: pd.DataFrame, idx: int) -> float:
    """ATR% für Triple-Barrier: gleitend, Fallback auf 1.5%."""
    try:
        import ta as ta_lib
        window = df.iloc[max(0, idx - 14): idx + 1]
        if len(window) < 5:
            return 0.015
        atr = float(ta_lib.volatility.average_true_range(
            window["high"], window["low"], window["close"], window=min(14, len(window))
        ).iloc[-1])
        price = float(df["close"].iloc[idx])
        return max(0.005, min(0.10, atr / price if price > 0 else 0.015))
    except Exception:
        return 0.015


def bootstrap_from_history(
    symbol: str,
    df: pd.DataFrame,
    store: MLDataStore,
    model: TradingModel,
):
    """Generiert Trainingsdaten aus historischen OHLCV-Daten und trainiert das Modell."""
    min_window = 60
    xs, ys = [], []

    for i in range(min_window, len(df) - LOOKFORWARD_H):
        window = df.iloc[: i + 1]
        try:
            feats = extract_features(window)
            atr_pct = _get_atr_pct(df, i)
            label = _compute_label_triple_barrier(df, i, atr_pct)
            xs.append(feats)
            ys.append(label)

            ts_val = df.index[i]
            ts     = int(ts_val.timestamp()) if hasattr(ts_val, "timestamp") else int(time.time()) - (len(df) - i) * 3600
            price  = float(df["close"].iloc[i])
            store.store(symbol, ts, feats, price, label)
            store.set_label(symbol, ts, label)
        except Exception:
            continue

    if len(xs) < model.MIN_SAMPLES:
        logger.warning("Bootstrap %s: nur %d Samples (min %d)", symbol, len(xs), model.MIN_SAMPLES)
        return

    X = np.array(xs, np.float32)
    y = np.array(ys, np.int32)
    model.train(X, y)
    logger.info("Bootstrap %s abgeschlossen: %d Samples", symbol, len(X))


class ModelTrainer:
    def __init__(self, store: MLDataStore, models: Dict[str, TradingModel]):
        self.store  = store
        self.models = models
        self._last_retrain_ts: Dict[str, int] = {}

    def record(self, symbol: str, ts: int, features: np.ndarray, price: float, predicted: int):
        self.store.store(symbol, ts, features, price, predicted)

    def label_and_maybe_retrain(self, symbol: str, current_df: pd.DataFrame):
        """Labelt gereifte Samples und retraint das Modell wenn genug neue Daten vorhanden."""
        if len(current_df) < LOOKFORWARD_H + 10:
            return

        cutoff_ts = int(time.time()) - LOOKFORWARD_H * 3600
        pending   = [
            (s, ts, p)
            for s, ts, p in self.store.get_unlabeled_before(cutoff_ts)
            if s == symbol
        ]
        if not pending:
            return

        labeled = 0
        for _, ts, _ in pending:
            target = pd.to_datetime(ts, unit="s", utc=True)
            idx    = int(current_df.index.get_indexer([target], method="nearest")[0])
            if idx < 0 or idx + LOOKFORWARD_H >= len(current_df):
                continue
            atr_pct = _get_atr_pct(current_df, idx)
            label = _compute_label_triple_barrier(current_df, idx, atr_pct)
            self.store.set_label(symbol, ts, label)
            labeled += 1

        if labeled:
            logger.info("Gelabelt %d neue Samples für %s", labeled, symbol)
            self._maybe_retrain(symbol)

    def _maybe_retrain(self, symbol: str):
        last_ts   = self._last_retrain_ts.get(symbol, 0)
        new_count = self.store.count_new_labeled_since(last_ts)
        if new_count < RETRAIN_EVERY_N:
            return

        X, y  = self.store.get_labeled(symbol)
        model = self.models.get(symbol)
        if model and len(X) >= model.MIN_SAMPLES:
            model.train(X, y)
            self._last_retrain_ts[symbol] = int(time.time())
            logger.info("Retrain %s: %d Samples gesamt", symbol, len(X))
