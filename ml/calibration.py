"""
Model quality metrics beyond simple F1.

Sharpe-based model selection: simulate what PnL would look like if you followed
the model's predictions on OOS data. A model that's "right" on big moves
is much more valuable than one with high accuracy on small moves.

Brier score and reliability data for calibration quality.
"""

import logging
import math
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Minimum annualized Sharpe on OOS simulation to accept a new model
MIN_OOS_SHARPE = 0.5


def simulated_sharpe(
    y_true: List[int],
    y_pred: List[int],
    returns: List[float],
    label_buy: int = 2,
    label_sell: int = 0,
) -> float:
    """
    Simulate PnL by taking the model's predicted direction × actual return.
    Returns annualized Sharpe ratio of the simulated strategy.

    y_true: actual labels (unused here, kept for API consistency)
    y_pred: predicted labels (0=sell, 1=hold, 2=buy)
    returns: forward returns for each OOS sample (fraction)
    label_buy / label_sell: integer codes from LABEL_TO_STR
    """
    if len(y_pred) != len(returns) or len(returns) < 10:
        return 0.0

    pnl = []
    for pred, ret in zip(y_pred, returns):
        if pred == label_buy:
            pnl.append(ret)
        elif pred == label_sell:
            pnl.append(-ret)
        # hold → no position, 0 PnL
        # (not appending keeps it out of mean/std calc — only active signals)

    if len(pnl) < 5:
        return 0.0

    arr = np.array(pnl)
    mean = arr.mean()
    std = arr.std()
    if std < 1e-9:
        return 0.0

    # Annualize: assuming samples are 1h candles → sqrt(24*365)
    sharpe = (mean / std) * math.sqrt(24 * 365)
    return float(sharpe)


def brier_score(y_true_bin: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Brier score for a single class (lower is better, 0=perfect, 1=worst).
    y_true_bin: binary array (1 = positive class)
    y_prob: predicted probability for positive class
    """
    return float(np.mean((y_prob - y_true_bin) ** 2))


def reliability_data(
    y_true_bin: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (mean_predicted_prob, fraction_positive) for a reliability diagram.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    mean_probs, fracs = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        mean_probs.append(float(y_prob[mask].mean()))
        fracs.append(float(y_true_bin[mask].mean()))
    return np.array(mean_probs), np.array(fracs)
