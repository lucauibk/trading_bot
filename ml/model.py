import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score

logger = logging.getLogger("ml.model")

MODEL_DIR = Path("data/models")

LABEL_TO_STR = {0: "sell", 1: "hold", 2: "buy"}
MIN_OOS_F1   = 0.30   # Modell wird nur gespeichert wenn OOS-F1 ≥ dieser Wert


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

        if oos_f1 < MIN_OOS_F1 and self._clf is not None:
            logger.warning(
                "OOS F1 %.3f < %.2f für %s – behalte altes Modell", oos_f1, MIN_OOS_F1, self.symbol
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

        self._clf = clf
        self._n_samples = len(X)
        self._last_oos_f1 = oos_f1
        self._save()

        classes, counts = np.unique(y, return_counts=True)
        dist = {LABEL_TO_STR.get(int(c), str(c)): int(n) for c, n in zip(classes, counts)}
        logger.info(
            "Modell gespeichert %s | %d Samples | OOS-F1=%.3f | Klassen: %s",
            self.symbol, len(X), oos_f1, dist,
        )

    def predict(self, x: np.ndarray) -> Tuple[int, float]:
        """Returns (label_int, calibrated_confidence). label: 0=sell, 1=hold, 2=buy"""
        if not self.is_ready():
            return 1, 0.0
        import pandas as pd
        from .features import FEATURE_NAMES
        x2 = pd.DataFrame(x.reshape(1, -1), columns=FEATURE_NAMES)
        proba = self._clf.predict_proba(x2)[0]
        label = int(proba.argmax())
        confidence = float(proba.max())
        return label, confidence

    def is_ready(self) -> bool:
        return self._clf is not None and self._n_samples >= self.MIN_SAMPLES

    def _save(self):
        joblib.dump(self._clf, self._model_path)
        self._meta_path.write_text(json.dumps({
            "n_samples": self._n_samples,
            "oos_f1": self._last_oos_f1,
        }))

    def _load(self):
        if not self._model_path.exists():
            return
        try:
            self._clf = joblib.load(self._model_path)
            if self._meta_path.exists():
                meta = json.loads(self._meta_path.read_text())
                self._n_samples  = meta.get("n_samples", self.MIN_SAMPLES)
                self._last_oos_f1 = meta.get("oos_f1", 0.0)
            else:
                self._n_samples = self.MIN_SAMPLES
            logger.info(
                "Modell geladen: %s (%d Samples, OOS-F1=%.3f)",
                self.symbol, self._n_samples, self._last_oos_f1,
            )
        except Exception as e:
            logger.warning("Modell-Ladefehler %s: %s – verwerfe altes Modell", self.symbol, e)
            self._clf = None
            self._n_samples = 0
