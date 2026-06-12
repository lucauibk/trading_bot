"""
MarketContext – shared state between all strategies.
Holds BTC regime, funding data, equity curve, and per-symbol positions.
Lock-protected for thread-safe access from the main loop and ML retraining threads.
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class FundingInfo:
    symbol: str
    rate: float          # current funding rate (e.g. 0.0001 = 0.01%)
    rate_z7d: float      # z-score vs 7-day history (positive = bullish pressure)
    oi_change_1h: float  # OI change last hour (%)
    oi_change_24h: float # OI change last 24h (%)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BTCContext:
    trend: str           # "up" | "down" | "range"
    return_1h: float     # BTC 1h return
    return_4h: float     # BTC 4h return
    return_24h: float    # BTC 24h return
    realized_vol_7d: float  # annualized realized vol
    dominance: float     # BTC dominance (0..1)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Position:
    symbol: str
    side: str            # "grid" | "directional"
    entry_price: float
    qty: float
    usdt_value: float
    leverage: float
    tp: Optional[float] = None
    sl: Optional[float] = None
    entry_ts: float = 0.0
    momentum_holds: int = 0


class MarketContext:
    def __init__(self):
        self._lock = threading.RLock()

        self.btc: Optional[BTCContext] = None
        self.funding: Dict[str, FundingInfo] = {}
        self.correlations: Dict[str, float] = {}  # symbol → rolling 30d BTC corr

        # Current equity across all coins
        self.total_equity: float = 0.0
        self.daily_start_equity: float = 0.0

        # Per-symbol positions (used by RiskManager)
        self.positions: Dict[str, List[Position]] = {}

        # Freeze flag (cross-coin daily drawdown brake)
        self.freeze_mode: bool = False

        # Stop flags (graceful shutdown)
        self.stop_mode: Optional[str] = None  # None | "sell_all" | "wait_fills"

    # ── Thread-safe accessors ─────────────────────────────────────────────────

    def set_btc(self, ctx: BTCContext):
        with self._lock:
            self.btc = ctx

    def get_btc(self) -> Optional[BTCContext]:
        with self._lock:
            return self.btc

    def set_funding(self, symbol: str, info: FundingInfo):
        with self._lock:
            self.funding[symbol] = info

    def get_funding(self, symbol: str) -> Optional[FundingInfo]:
        with self._lock:
            return self.funding.get(symbol)

    def set_correlation(self, symbol: str, corr: float):
        with self._lock:
            self.correlations[symbol] = corr

    def get_correlation(self, symbol: str) -> float:
        with self._lock:
            return self.correlations.get(symbol, 0.0)

    def add_position(self, pos: Position):
        with self._lock:
            self.positions.setdefault(pos.symbol, []).append(pos)

    def remove_position(self, symbol: str, side: str):
        with self._lock:
            self.positions[symbol] = [
                p for p in self.positions.get(symbol, []) if p.side != side
            ]

    def open_position_count(self) -> int:
        with self._lock:
            return sum(len(v) for v in self.positions.values())

    def get_positions(self, symbol: str) -> list:
        with self._lock:
            return list(self.positions.get(symbol, []))

    def symbol_position_usdt(self, symbol: str) -> float:
        with self._lock:
            return sum(p.usdt_value for p in self.positions.get(symbol, []))

    def set_equity(self, equity: float):
        with self._lock:
            self.total_equity = equity

    def set_freeze(self, freeze: bool):
        with self._lock:
            self.freeze_mode = freeze

    def is_frozen(self) -> bool:
        with self._lock:
            return self.freeze_mode

    def set_stop_mode(self, mode: Optional[str]):
        with self._lock:
            self.stop_mode = mode

    def get_stop_mode(self) -> Optional[str]:
        with self._lock:
            return self.stop_mode
