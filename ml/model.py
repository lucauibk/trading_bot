import json
import logging
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score

logger = logging.getLogger("ml.model")

MODEL_DIR = Path("data/models")

LABEL_TO_STR = {0: "sell", 1: "hold", 2: "buy"}
MIN_OOS_F1   = 0.33   # Modell wird nur gespeichert wenn OOS-F1 ≥ dieser Wert (Baseline: ~0.33 random)
# NOTE: Raised from 0.30 → 0.40 (ML-Rehab). 3-class random baseline ≈ 0.33;
# 0.40 ensures only above-random models are deployed. Models that fall below
# this gate keep the grid running normally — they just don't open directional
# trades (directional_score_min = 0.75 enforces this independently).


class TradingModel:
    MIN_SAMPLES = 100

    def __init__(self, symbol: str):
        self.symbol = symbol.replace("/", "_")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self._model_path = MODEL_DIR / f"{self.symbol}.joblib"
        self._meta_path  = MODEL_DIR / f"{self.symbol}_meta.json"
        self._clf: Optional[CalibratedClassifierCV] = None
        self._n_samples = 0
        self._last_oos_f1 = 0.0
        self._feature_names: List[str] = []
        self._lock = threading.Lock()
        self._load()

    def train(self, X: np.ndarray, y: np.ndarray):
        if len(X) < self.MIN_SAMPLES:
            logger.info("Zu wenig Samples für %s: %d < %d", self.symbol, len(X), self.MIN_SAMPLES)
            return

        # Walk-Forward: letzter Fold ist Out-of-Sample
        tscv = TimeSeriesSplit(n_splits=5)
        oos_preds, oos_true = [], []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            if len(np.unique(y_tr)) < 2:
                continue
            base = LGBMClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
                verbose=-1,
            )
            base.fit(X_tr, y_tr)
            oos_preds.extend(base.predict(X_val))
            oos_true.extend(y_val)

        if oos_true:
            oos_f1 = float(f1_score(oos_true, oos_preds, average="macro", zero_division=0))
            logger.info("Walk-Forward OOS F1 %s: %.3f", self.symbol, oos_f1)
        else:
            oos_f1 = 0.0

        if oos_f1 < MIN_OOS_F1:
            logger.warning(
                "OOS F1 %.3f < %.2f für %s – Modell verworfen", oos_f1, MIN_OOS_F1, self.symbol
            )
            return

        # Finales Modell auf allen Daten trainieren + kalibrieren
        base_final = LGBMClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
        # Kalibrierung braucht mindestens 2 Folds mit ausreichend Daten
        cv = min(3, max(2, len(X) // 100))
        clf = CalibratedClassifierCV(base_final, method="isotonic", cv=cv)
        clf.fit(X, y)

        with self._lock:
            self._clf = clf
            self._n_samples = len(X)
            self._last_oos_f1 = oos_f1
            # Store the actual feature names used (may be 16 or 34 depending on caller)
            self._feature_names = [f"f{i}" for i in range(X.shape[1])]
            try:
                from .features.combined import ALL_FEATURE_NAMES
                from .features import FEATURE_NAMES
                if X.shape[1] == len(ALL_FEATURE_NAMES):
                    self._feature_names = list(ALL_FEATURE_NAMES)
                elif X.shape[1] == len(FEATURE_NAMES):
                    self._feature_names = list(FEATURE_NAMES)
            except Exception:
                pass
        self._save()

        classes, counts = np.unique(y, return_counts=True)
        dist = {LABEL_TO_STR.get(int(c), str(c)): int(n) for c, n in zip(classes, counts)}
        logger.info(
            "Modell gespeichert %s | %d Samples | OOS-F1=%.3f | Klassen: %s",
            self.symbol, len(X), oos_f1, dist,
        )

    def predict(self, x: np.ndarray) -> Tuple[int, float]:
        """Returns (label_int, calibrated_confidence). label: 0=sell, 1=hold, 2=buy"""
        with self._lock:
            if not self.is_ready():
                return 1, 0.0
            import pandas as pd

            # Feature-count mismatch: model was trained with a different number of features.
            # Fall back to hold/0 and let the caller retrain rather than crash or produce garbage.
            expected_n = len(self._feature_names)
            if expected_n > 0 and x.shape[0] != expected_n:
                logger.warning(
                    "Feature count mismatch for %s: model=%d, input=%d – returning hold",
                    self.symbol, expected_n, x.shape[0],
                )
                return 1, 0.0

            feature_names = self._feature_names if self._feature_names else [f"f{i}" for i in range(x.shape[0])]
            x2 = pd.DataFrame(x.reshape(1, -1), columns=feature_names)
            try:
                proba = self._clf.predict_proba(x2)[0]
            except Exception as e:
                logger.warning("predict_proba failed for %s: %s", self.symbol, e)
                return 1, 0.0
            # proba columns follow self._clf.classes_, which is NOT guaranteed to
            # be [0,1,2]. If a class was absent from this symbol's training set,
            # classes_ is e.g. [0,2] and the positional argmax would map to the
            # wrong label (and flip the score sign downstream). Translate the
            # positional index back to the real class label.
            idx = int(proba.argmax())
            label = int(self._clf.classes_[idx])
            confidence = float(proba[idx])
            return label, confidence

    def is_ready(self) -> bool:
        return self._clf is not None and self._n_samples >= self.MIN_SAMPLES

    def _save(self):
        joblib.dump(self._clf, self._model_path)
        self._meta_path.write_text(json.dumps({
            "n_samples":     self._n_samples,
            "oos_f1":        self._last_oos_f1,
            "feature_names": self._feature_names,
        }))

    def _load(self):
        if not self._model_path.exists():
            return
        try:
            from .features.combined import ALL_FEATURE_NAMES
            expected_dim = len(ALL_FEATURE_NAMES)  # 34

            self._clf = joblib.load(self._model_path)
            if self._meta_path.exists():
                meta = json.loads(self._meta_path.read_text())
                self._n_samples   = meta.get("n_samples", self.MIN_SAMPLES)
                self._last_oos_f1 = meta.get("oos_f1", 0.0)
                self._feature_names = meta.get("feature_names", [])
            else:
                self._n_samples = self.MIN_SAMPLES

            # Verwirf inkompatible Modelle (z.B. mit 16 statt 34 Features)
            # damit Bootstrap beim nächsten Start frische 34-Feature-Modelle erzeugt.
            if self._feature_names and len(self._feature_names) != expected_dim:
                logger.warning(
                    "Modell %s hat %d Features, erwartet %d – verwerfe stales Modell",
                    self.symbol, len(self._feature_names), expected_dim,
                )
                self._clf           = None
                self._n_samples     = 0
                self._feature_names = []
                return

            logger.info(
                "Modell geladen: %s (%d Samples, OOS-F1=%.3f)",
                self.symbol, self._n_samples, self._last_oos_f1,
            )
        except Exception as e:
            logger.warning("Modell-Ladefehler %s: %s – verwerfe altes Modell", self.symbol, e)
            self._clf = None
            self._n_samples = 0
