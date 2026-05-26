"""
Backward-compatible re-export of extract_features for code that still imports
from ml.features directly (ml/predictor.py, ml/trainer.py).
"""

from ml.features.technical import extract as _extract_tech, FEATURE_NAMES, FEATURE_NAMES as FEATURE_NAMES
import numpy as np


def extract_features(df) -> np.ndarray:
    return _extract_tech(df)


N_FEATURES = len(FEATURE_NAMES)
