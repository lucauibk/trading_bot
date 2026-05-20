import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger("ml.model")

MODEL_DIR = Path("data/models")

LABEL_TO_STR = {0: "sell", 1: "hold", 2: "buy"}


class TradingModel:
    MIN_SAMPLES = 100

    def __init__(self, symbol: str):
        self.symbol = symbol.replace("/", "_")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self._model_path = MODEL_DIR / f"{self.symbol}.joblib"
        self._meta_path  = MODEL_DIR / f"{self.symbol}_meta.json"
        self._clf: Optional[RandomForestClassifier] = None
        self._n_samples = 0
        self._load()

    def train(self, X: np.ndarray, y: np.ndarray):
        if len(X) < self.MIN_SAMPLES:
            logger.info("Zu wenig Samples für %s: %d < %d", self.symbol, len(X), self.MIN_SAMPLES)
            return
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        clf.fit(X, y)
        self._clf = clf
        self._n_samples = len(X)
        self._save()
        classes, counts = np.unique(y, return_counts=True)
        dist = {LABEL_TO_STR.get(int(c), str(c)): int(n) for c, n in zip(classes, counts)}
        logger.info("Modell trainiert %s | %d Samples | Klassen: %s", self.symbol, len(X), dist)

    def predict(self, x: np.ndarray) -> Tuple[int, float]:
        """Returns (label_int, confidence).  label: 0=sell, 1=hold, 2=buy"""
        if not self.is_ready():
            return 1, 0.0
        x2 = x.reshape(1, -1)
        label      = int(self._clf.predict(x2)[0])
        confidence = float(self._clf.predict_proba(x2)[0].max())
        return label, confidence

    def is_ready(self) -> bool:
        return self._clf is not None and self._n_samples >= self.MIN_SAMPLES

    def _save(self):
        joblib.dump(self._clf, self._model_path)
        self._meta_path.write_text(json.dumps({"n_samples": self._n_samples}))

    def _load(self):
        if not self._model_path.exists():
            return
        try:
            self._clf = joblib.load(self._model_path)
            if self._meta_path.exists():
                meta = json.loads(self._meta_path.read_text())
                self._n_samples = meta.get("n_samples", self.MIN_SAMPLES)
            else:
                self._n_samples = self.MIN_SAMPLES
            logger.info("Modell geladen: %s (%d Samples)", self.symbol, self._n_samples)
        except Exception as e:
            logger.warning("Modell-Ladefehler %s: %s", self.symbol, e)
            self._clf = None
            self._n_samples = 0
