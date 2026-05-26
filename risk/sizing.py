"""
Position sizing: Fractional Kelly + Vol-Targeting.

Kelly formula: f = (p * b - q) / b
  where p = win probability, b = avg_win/avg_loss ratio, q = 1 - p

We apply a Kelly fraction (default 25%) for safety and cap at max_position_pct.
Vol-targeting: target 1% equity risk per trade, scale by realized volatility.
"""

import math


def kelly_fraction(win_rate: float, win_loss_ratio: float, kelly_factor: float = 0.25) -> float:
    """
    win_rate: fraction of trades that win (0..1)
    win_loss_ratio: avg_win / avg_loss
    kelly_factor: fraction of full Kelly to use (0.25 = quarter-Kelly)
    Returns fraction of equity to risk (0..1), capped at kelly_factor.
    """
    if win_loss_ratio <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.01
    q = 1.0 - win_rate
    full_kelly = (win_rate * win_loss_ratio - q) / win_loss_ratio
    return max(0.005, min(kelly_factor, full_kelly * kelly_factor))


def vol_target_size(equity: float, target_risk_pct: float, realized_vol: float,
                    price: float, leverage: float = 1.0) -> float:
    """
    Returns qty (in base asset units) so that 1-day P&L std ≈ target_risk_pct * equity.

    equity: total equity in USDT
    target_risk_pct: e.g. 0.01 = 1% daily equity risk
    realized_vol: daily realized volatility of the asset (e.g. 0.03 = 3%)
    price: current asset price
    leverage: leverage multiplier
    """
    if realized_vol <= 0 or price <= 0:
        return 0.0
    dollar_risk = equity * target_risk_pct
    # dollar_risk = qty * price * realized_vol * leverage
    qty = dollar_risk / (price * realized_vol * leverage)
    return max(0.0, qty)


def compute_position_usdt(
    equity: float,
    win_rate: float,
    win_loss_ratio: float,
    realized_vol: float,
    target_risk_pct: float = 0.01,
    max_position_pct: float = 0.10,
    kelly_factor: float = 0.25,
) -> float:
    """
    Returns recommended USDT position size.
    Uses the minimum of Kelly-sized and vol-targeted position.
    """
    kelly_pct = kelly_fraction(win_rate, win_loss_ratio, kelly_factor)
    kelly_usdt = equity * kelly_pct

    vol_usdt = equity * target_risk_pct / max(realized_vol, 0.005)

    sized = min(kelly_usdt, vol_usdt)
    capped = min(sized, equity * max_position_pct)
    return max(capped, equity * 0.005)  # floor at 0.5% equity
