import os
from dotenv import load_dotenv

load_dotenv()

# ── Exchange ──────────────────────────────────────────────────────────────────
EXCHANGE_ID = "kraken"
KRAKEN_FEE = 0.0016  # 0.16% Maker-Fee – einzige Quelle, überall von hier importieren (#53)
API_KEY = os.getenv("KRAKEN_API_KEY") or os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("KRAKEN_API_SECRET") or os.getenv("BINANCE_API_SECRET", "")
PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() == "true"

# ── Märkte & Timeframe ────────────────────────────────────────────────────────
SYMBOLS = ["SOL/USD", "ETH/USD", "DOT/USD", "LINK/USD"]
TIMEFRAME = "1h"
LOOKBACK_CANDLES = 300        # Candles die initial geladen werden

# ── Strategie-Parameter (RSI Mean Reversion) ─────────────────────────────────
EMA_TREND = 50                # Trendfilter: nur Long wenn Preis > EMA 50
RSI_PERIOD = 7                # Kürzere Periode = sensitiver
RSI_OVERSOLD = 30             # Optimaler Wert
RSI_EXIT = 65                 # RSI-basierter Exit
BB_PERIOD = 20
BB_STD = 2.0
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 1.5     # Stop-Loss = 1.5× ATR
ATR_TP_MULTIPLIER = 2.0       # TP = 2× ATR (R:R = 1:1.33)

# Dummy-Werte damit alter Code nicht bricht
EMA_FAST = 50
EMA_SLOW = 200
RSI_OVERBOUGHT = 70

# ── Risikomanagement ──────────────────────────────────────────────────────────
RISK_PER_TRADE = 0.02         # 2% des Kapitals pro Trade
MAX_OPEN_POSITIONS = 3
MIN_VOLUME_USDT = 1_000_000   # mind. 1M USDT Volumen in letzter Stunde

# ── Kapital ───────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "1000"))

# ── Telegram ──────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Bot Loop ──────────────────────────────────────────────────────────────────
CHECK_INTERVAL_SECONDS = 60   # Alle 60s nach neuen Candles schauen
