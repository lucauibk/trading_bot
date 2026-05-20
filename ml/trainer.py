import logging
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .data_store import MLDataStore
from .features import extract_features
from .model import TradingModel

logger = logging.getLogger("ml.trainer")

LOOKFORWARD_H  = 6      # 1h-Candles vorausschauen für Label-Berechnung
THRESHOLD      = 0.015  # 1.5% Preisbewegung = gerichtetes Signal
RETRAIN_EVERY_N = 50    # Retrain nach N neuen gelabelten Samples


def _compute_label(close: pd.Series, idx: int) -> int:
    """
    0=sell, 1=hold, 2=buy basierend auf den nächsten LOOKFORWARD_H Candles.
    Wenn beide Schwellen überschritten werden, gewinnt der erste Treffer.
    """
    if idx + LOOKFORWARD_H >= len(close):
        return 1
    p0     = float(close.iloc[idx])
    future = close.iloc[idx + 1: idx + 1 + LOOKFORWARD_H]
    max_ret = (float(future.max()) - p0) / p0
    min_ret = (float(future.min()) - p0) / p0
    up = max_ret >  THRESHOLD
    dn = min_ret < -THRESHOLD

    if up and not dn:
        return 2
    if dn and not up:
        return 0
    if up and dn:
        up_first = int((future >= p0 * (1 + THRESHOLD)).values.argmax())
        dn_first = int((future <= p0 * (1 - THRESHOLD)).values.argmax())
        return 2 if up_first <= dn_first else 0
    return 1


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
            label = _compute_label(df["close"], i)
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
            label = _compute_label(current_df["close"], idx)
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
