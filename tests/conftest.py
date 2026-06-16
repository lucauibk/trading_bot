import os

# Tests instantiate real GridStrategy objects and trigger real fills/SL events.
# Without this, those calls write directly into the production data/trades.db
# and fire real Telegram notifications (see strategies/grid.py's GRIDBOT_BACKTEST
# checks, the same flag backtest/engine.py sets for backtests/sweeps).
os.environ["GRIDBOT_BACKTEST"] = "1"
