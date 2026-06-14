"""
GridParams – tunable strategy parameters as an immutable dataclass.

Defaults reproduce the legacy hardcoded behaviour exactly, so a
GridStrategy constructed without params is bit-identical to before.
The backtest engine and the sweep script override fields via from_dict().
"""

from dataclasses import dataclass, field, fields, replace, asdict
from typing import Tuple


@dataclass(frozen=True)
class GridParams:
    # ── Stop-loss ──────────────────────────────────────────────────────
    # "per_position": SL a fixed pct below each buy (legacy)
    # "floor":        SL below the grid's lower bound — nothing inside the
    #                 grid stops out, only a range breakdown does.
    #                 Cascade-safe: at most one flush per grid rebuild.
    sl_mode: str = "floor"                 # changed from "per_position" to prevent cascades
    floor_sl_atr_mult: float = 1.0         # floor = grid_lower − mult × ATR
    per_pos_sl_step_mult: float = 1.5      # per_position mode: SL = step_pct × mult below buy
    per_pos_sl_min_pct: float = 0.008      # per_position mode: SL floor 0.8% below buy
    per_pos_sl_max_pct: float = 0.04       # per_position mode: SL hard-cap 4% below buy
    momentum_hold_score: float = 0.35      # delay SL while score above this …
    momentum_hold_max: int = 2             # … at most this many ticks

    # ── Grid geometry ──────────────────────────────────────────────────
    levels_by_regime: Tuple[Tuple[str, int], ...] = (
        ("ranging", 14), ("trending", 6), ("volatile", 20)
    )
    range_atr_mult_trending: float = 2.0
    range_atr_mult_volatile: float = 1.5
    min_step_pct: float = 0.0              # 0 = off; else cap levels so step ≥ this

    # ── Trend filter (ML-independent) ──────────────────────────────────
    # Default ON: sweep winner 2026-06-12 (results/sweep_20260612_1242) —
    # every top-8 OOS config had the filter active; it roughly halved
    # worst-case drawdown across symbols.
    trend_filter_enabled: bool = True
    trend_adx_min: float = 25.0

    # ── Leverage ───────────────────────────────────────────────────────
    leverage: float = 0.0                  # 0 = read live from dashboard DB

    # ── Directional trades ─────────────────────────────────────────────
    directional_enabled: bool = True
    directional_score_min: float = 0.12
    directional_pct: float = 0.20
    directional_tp_atr: float = 3.0
    directional_sl_atr: float = 1.5

    @property
    def regime_levels(self) -> dict:
        return dict(self.levels_by_regime)

    @classmethod
    def from_dict(cls, d: dict) -> "GridParams":
        """Build from a plain dict, ignoring unknown keys.

        levels_by_regime may be given as a dict ({"ranging": 14, ...}).
        """
        known = {f.name for f in fields(cls)}
        kwargs = {}
        for k, v in d.items():
            if k not in known:
                continue
            if k == "levels_by_regime" and isinstance(v, dict):
                v = tuple(sorted(v.items()))
            elif k == "levels_by_regime" and isinstance(v, list):
                v = tuple(tuple(pair) for pair in v)
            kwargs[k] = v
        return cls(**kwargs)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["levels_by_regime"] = dict(self.levels_by_regime)
        return d
