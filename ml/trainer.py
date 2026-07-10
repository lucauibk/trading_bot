import logging
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .data_store import MLDataStore
from .features.combined import extract_all as extract_all_features
from .model import TradingModel

logger = logging.getLogger("ml.trainer")

LOOKFORWARD_H   = 2      # Timeout in Candles (2h) – grid trades are micro-moves
RETRAIN_EVERY_N = 50     # Retrain nach N neuen gelabelten Samples

# Triple-Barrier-Parameter: eng genug für Grid-Mikro-Moves (0.1–0.5% per trade)
TB_UPPER_ATR = 0.5   # Gewinn-Barriere = 0.5× ATR  (war 2.0× → zu weit für Grid)
TB_LOWER_ATR = 0.5   # Verlust-Barriere = 0.5× ATR (war 1.5× → Swing-Trade-Skala)


def _compute_rolling_btc_corr(df: pd.DataFrame, btc_df: pd.DataFrame, window: int = 720) -> "pd.Series":
    """
    Pre-computes a rolling 30-day (720 × 1h candles) Pearson correlation between
    symbol and BTC log-returns. Aligned on the symbol's index.

    Returned Series is indexed by df.index; NaN where insufficient history.
    Caller is responsible for handling NaN (fall back to 0.0).
    """
    sym_ret = np.log(df["close"] / df["close"].shift(1))
    btc_ret = np.log(btc_df["close"] / btc_df["close"].shift(1))
    # Align: inner join on datetime index, then compute rolling corr on symbol axis
    aligned = pd.DataFrame({"sym": sym_ret, "btc": btc_ret}).dropna()
    if len(aligned) < 60:
        return pd.Series(dtype=float)
    corr = aligned["sym"].rolling(window, min_periods=60).corr(aligned["btc"])
    # Reindex back to the full symbol index (fills gaps with NaN)
    return corr.reindex(df.index)


def _extract_training_features(df: pd.DataFrame, window: pd.DataFrame,
                               btc_corr: float = 0.0) -> Optional[np.ndarray]:
    """
    Extract the 34-feature vector for a training window.
    htf(4) and seasonality(5) are computable from OHLCV + timestamp.
    market(5) uses aligned BTC returns from the same df (historical backfill).
    perp(4) defaults to 0 (no historical funding-rate archive available).
    btc_corr: rolling 30d BTC-correlation passed from bootstrap/refresh caller.

    Returns None (and logs the cause) if extraction fails, so the caller can skip
    the sample. We deliberately do NOT fall back to a 16-feature technical vector:
    that silently violates the 34-feature contract (CLAUDE.md) and yields either an
    np.array(dtype) crash on mixed dims or a 16-feature model that TradingModel._load()
    later discards as stale — leaving the bot with no usable model and no error (#55).
    """
    try:
        dt = window.index[-1].to_pydatetime() if hasattr(window.index[-1], "to_pydatetime") else None
        return extract_all_features(window, funding=None, btc=None, btc_corr=btc_corr, dt=dt)
    except Exception:
        logger.debug("34-feature extraction failed for a training sample – skipping",
                     exc_info=True)
        return None


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
    btc_df: Optional[pd.DataFrame] = None,
):
    """Generiert Trainingsdaten aus historischen OHLCV-Daten und trainiert das Modell.

    btc_df: optional BTC/USD OHLCV aligned to the same timeframe and covering the
    same range. When provided the trainer computes a rolling 30d BTC-correlation
    feature (btc_corr_30d) instead of the constant 0.0 placeholder.
    """
    min_window = 60
    xs, ys = [], []
    skipped_feat = 0

    # Pre-compute rolling BTC-correlation series if BTC data is available
    btc_corr_series = None
    if btc_df is not None and len(btc_df) >= 60:
        try:
            btc_corr_series = _compute_rolling_btc_corr(df, btc_df)
        except Exception as exc:
            logger.warning("btc_corr pre-compute failed for %s: %s", symbol, exc)

    for i in range(min_window, len(df) - LOOKFORWARD_H):
        window = df.iloc[: i + 1]
        # Resolve per-step BTC correlation (NaN → 0.0 fallback)
        btc_corr = 0.0
        if btc_corr_series is not None and i < len(btc_corr_series):
            val = btc_corr_series.iloc[i]
            if not (isinstance(val, float) and val != val):  # not NaN
                btc_corr = float(np.clip(val, -1.0, 1.0))
        try:
            feats = _extract_training_features(df, window, btc_corr=btc_corr)
            if feats is None:            # 34-feature extraction failed → skip, keep dims uniform
                skipped_feat += 1
                continue
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

    if skipped_feat:
        logger.warning("Bootstrap %s: %d Samples wegen fehlgeschlagener 34-Feature-"
                       "Extraktion übersprungen (Traceback auf DEBUG)", symbol, skipped_feat)

    if len(xs) < model.MIN_SAMPLES:
        logger.warning("Bootstrap %s: nur %d Samples (min %d)", symbol, len(xs), model.MIN_SAMPLES)
        return

    X = np.array(xs, np.float32)
    y = np.array(ys, np.int32)
    model.train(X, y)
    logger.info("Bootstrap %s abgeschlossen: %d Samples", symbol, len(X))


def refresh_from_recent_history(
    symbol: str,
    df: pd.DataFrame,
    store: MLDataStore,
    model: TradingModel,
    btc_df: Optional[pd.DataFrame] = None,
):
    """
    Täglicher ML-Refresh auf frischen OHLCV-Daten (letzten 30 Tage).
    Ersetzt das Modell nur wenn neues OOS-F1 ≥ altes F1 - 0.05.
    LLM-frei — reiner LightGBM-Retrain.

    btc_df: optional BTC/USD OHLCV for rolling btc_corr_30d feature (same fix as bootstrap).
    """
    min_window = 60
    xs, ys = [], []
    skipped_feat = 0

    btc_corr_series = None
    if btc_df is not None and len(btc_df) >= 60:
        try:
            btc_corr_series = _compute_rolling_btc_corr(df, btc_df)
        except Exception as exc:
            logger.warning("btc_corr pre-compute failed for %s (refresh): %s", symbol, exc)

    for i in range(min_window, len(df) - LOOKFORWARD_H):
        window = df.iloc[: i + 1]
        btc_corr = 0.0
        if btc_corr_series is not None and i < len(btc_corr_series):
            val = btc_corr_series.iloc[i]
            if not (isinstance(val, float) and val != val):
                btc_corr = float(np.clip(val, -1.0, 1.0))
        try:
            feats = _extract_training_features(df, window, btc_corr=btc_corr)
            if feats is None:            # 34-feature extraction failed → skip, keep dims uniform
                skipped_feat += 1
                continue
            atr_pct = _get_atr_pct(df, i)
            label = _compute_label_triple_barrier(df, i, atr_pct)
            xs.append(feats)
            ys.append(label)
        except Exception:
            continue

    if skipped_feat:
        logger.warning("Refresh %s: %d Samples wegen fehlgeschlagener 34-Feature-"
                       "Extraktion übersprungen (Traceback auf DEBUG)", symbol, skipped_feat)

    if len(xs) < model.MIN_SAMPLES:
        logger.warning("Refresh %s: nur %d Samples – übersprungen", symbol, len(xs))
        return

    old_f1 = model._last_oos_f1
    # Preserve the current classifier so we can roll back if quality drops
    old_clf         = model._clf
    old_n_samples   = model._n_samples

    X = np.array(xs, np.float32)
    y = np.array(ys, np.int32)

    # train() has Walk-Forward + F1-Guard (MIN_OOS_F1=0.30) built in.
    # Additional rollback: restore previous model if new F1 is notably worse.
    model.train(X, y)
    new_f1 = model._last_oos_f1

    if new_f1 < old_f1 - 0.05 and old_clf is not None:
        logger.warning(
            "[ML REFRESH] %s: F1 %.3f → %.3f – rolling back to previous model",
            symbol, old_f1, new_f1,
        )
        with model._lock:
            model._clf          = old_clf
            model._n_samples    = old_n_samples
            model._last_oos_f1  = old_f1
        model._save()
    else:
        logger.info(
            "[ML REFRESH] %s: F1 %.3f → %.3f – akzeptiert (%d Samples)",
            symbol, old_f1, new_f1, len(xs),
        )


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
            # get_indexer(method="nearest") always clamps to a valid in-range
            # index for a non-empty index (it never returns -1), so a sample whose
            # ts predates the ~120-candle window gets pinned to index 0 and would be
            # mislabeled from the *first* candle's forward window — and that wrong
            # label is then persisted permanently. Reject matches that aren't within
            # one candle (3600s) of the sample's real timestamp. (#91)
            if abs((current_df.index[idx] - target).total_seconds()) > 3600:
                continue
            atr_pct = _get_atr_pct(current_df, idx)
            label = _compute_label_triple_barrier(current_df, idx, atr_pct)
            self.store.set_label(symbol, ts, label)
            labeled += 1

        if labeled:
            logger.info("Gelabelt %d neue Samples für %s", labeled, symbol)
            self._maybe_retrain(symbol)

    def _maybe_retrain(self, symbol: str):
        try:
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
            elif model:
                logger.debug(
                    "Retrain %s übersprungen: nur %d saubere 34-Feature-Samples (min %d)",
                    symbol, len(X), model.MIN_SAMPLES,
                )
        except Exception as e:
            logger.error("Retrain fehlgeschlagen %s: %s", symbol, e, exc_info=True)
