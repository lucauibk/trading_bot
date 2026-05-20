"""
Abstrakte Basisklasse für alle Strategien.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd


class Signal(Enum):
    LONG  = "LONG"
    SHORT = "SHORT"
    NONE  = "NONE"


@dataclass
class TradeSignal:
    signal:     Signal
    entry:      float
    stop_loss:  float
    take_profit: float
    reason:     str = ""

    @property
    def is_valid(self) -> bool:
        return self.signal != Signal.NONE and self.stop_loss > 0


class BaseStrategy(ABC):

    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Analysiert den DataFrame und gibt ein TradeSignal zurück."""

    @abstractmethod
    def name(self) -> str:
        """Kurzname der Strategie."""

    def get_stop_loss(self, entry: float, signal: Signal, atr: float,
                      mult: Optional[float] = None) -> float:
        mult = mult or self.params.get("atr_stop_mult", 1.5)
        if signal == Signal.LONG:
            return entry - mult * atr
        return entry + mult * atr

    def get_take_profit(self, entry: float, stop_loss: float,
                        rr: Optional[float] = None) -> float:
        rr = rr or self.params.get("rr_ratio", 2.0)
        risk = abs(entry - stop_loss)
        return entry + rr * risk
