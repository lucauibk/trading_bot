# Sweep Report

Symbole: SOL/USD, ETH/USD, AVAX/USD, LINK/USD, XRP/USD | 180d (Train 120d / Test 60d) | Leverage 3.0×

## Top-Configs: Train vs. OOS (Ranking nach OOS-median-Calmar)

| # | cfg | OOS Calmar | OOS Ret% | OOS worstDD | Train Calmar | Train Ret% | Params |
|---|-----|-----------|----------|-------------|--------------|------------|--------|
| 1 | 13 | 103.12 | 40.5 | -8.6 | 305.83 | 174.6 | `{"sl_mode": "per_position", "momentum_hold_max": 2, "trend_filter_enabled": true, "min_step_pct": 0.0, "directional_enabled": true}` |
| 2 | 5 | 95.69 | 40.5 | -8.6 | 322.25 | 181.0 | `{"sl_mode": "per_position", "momentum_hold_max": 0, "trend_filter_enabled": true, "min_step_pct": 0.0, "directional_enabled": true}` |
| 3 | 12 | 94.11 | 39.7 | -8.6 | 326.94 | 180.7 | `{"sl_mode": "per_position", "momentum_hold_max": 2, "trend_filter_enabled": true, "min_step_pct": 0.0, "directional_enabled": false}` |
| 4 | 7 | 90.36 | 38.5 | -11.1 | 390.06 | 214.2 | `{"sl_mode": "per_position", "momentum_hold_max": 0, "trend_filter_enabled": true, "min_step_pct": 0.01, "directional_enabled": true}` |
| 5 | 15 | 88.89 | 37.7 | -11.1 | 374.17 | 210.5 | `{"sl_mode": "per_position", "momentum_hold_max": 2, "trend_filter_enabled": true, "min_step_pct": 0.01, "directional_enabled": true}` |
| 6 | 4 | 87.19 | 39.7 | -8.6 | 344.64 | 187.3 | `{"sl_mode": "per_position", "momentum_hold_max": 0, "trend_filter_enabled": true, "min_step_pct": 0.0, "directional_enabled": false}` |
| 7 | 6 | 87.08 | 36.3 | -11.1 | 418.75 | 221.4 | `{"sl_mode": "per_position", "momentum_hold_max": 0, "trend_filter_enabled": true, "min_step_pct": 0.01, "directional_enabled": false}` |
| 8 | 14 | 85.66 | 35.4 | -11.1 | 399.58 | 217.6 | `{"sl_mode": "per_position", "momentum_hold_max": 2, "trend_filter_enabled": true, "min_step_pct": 0.01, "directional_enabled": false}` |
| 9 | 8 | 46.99 | 33.3 | -14.9 | 180.21 | 137.1 | `{"sl_mode": "per_position", "momentum_hold_max": 2, "trend_filter_enabled": false, "min_step_pct": 0.0, "directional_enabled": false}` |
| 10 | 0 | 43.08 | 30.9 | -14.4 | 188.98 | 142.6 | `{"sl_mode": "per_position", "momentum_hold_max": 0, "trend_filter_enabled": false, "min_step_pct": 0.0, "directional_enabled": false}` |