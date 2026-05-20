"""
Grid Trading Bot – generiert kleine Gewinne bei Auf-und-Ab Bewegungen.
Aufruf: python3 grid_bot.py

Funktioniert am besten wenn der Markt seitwärts läuft.
"""

import threading
import time
import logging
import signal
import sys
from typing import Optional

import ta as ta_lib
import config
import notifier
from data_fetcher import fetch_ticker, fetch_ohlcv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/grid_bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("grid_bot")

# ── Grid Parameter ────────────────────────────────────────────────────────────
GRIDS = [
    {"symbol": "SOL/USD",  "investment": 300.0, "levels": 8},
    {"symbol": "LINK/USD", "investment": 300.0, "levels": 8},
    {"symbol": "DOT/USD",  "investment": 300.0, "levels": 8},
    {"symbol": "ETH/USD",  "investment": 300.0, "levels": 8},
    {"symbol": "DOGE/USD", "investment": 300.0, "levels": 8},
]
CHECK_INTERVAL   = 15

# ── Dynamisches Grid (Volatilität) ────────────────────────────────────────────
GRID_RANGE_MIN   = 0.03    # Mindest-Range (3% = 0.5% Schritt × 6 Level – deckt Live-Gebühren)
GRID_RANGE_MAX   = 0.25    # Maximum-Range bei sehr volatiler Markt
ATR_CANDLES      = 24     # ATR über letzte 24h berechnen

# ── Notbremse ─────────────────────────────────────────────────────────────────
MAX_LOSS_PCT         = 0.08   # Pro Coin – 8% des aktuellen Investments → skaliert mit Compounding
PER_POS_SL_PCT       = 0.04   # Per-Position Stop-Loss: 4% unter Buy-Preis
MAX_INVESTMENT_MULT  = 3.0    # Compounding-Cap: max. 3× Initial-Investment pro Coin

# ── Directional Trades (KI kauft aktiv bei UP-Signal) ─────────────────────────
DIRECTIONAL_ENABLED   = True
DIRECTIONAL_SCORE_MIN = 0.15   # Score > 0.15 bei UP-Signal → kaufen
DIRECTIONAL_PCT       = 0.15   # 15% des Investments pro Directional Trade
DIRECTIONAL_TP_ATR    = 2.5    # Take-Profit: Einstieg + 2.5 × ATR
DIRECTIONAL_SL_ATR    = 1.5    # Stop-Loss:   Einstieg − 1.5 × ATR

# ── Graceful Shutdown ─────────────────────────────────────────────────────────
_running = True
def _shutdown(sig, frame):
    global _running
    _running = False
    logger.info("Shutdown…")

signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


COMPOUND_EVERY_TRADES = 5      # Nach jeweils X Trades wird Gewinn reinvestiert

# ── Vorhersage ────────────────────────────────────────────────────────────────
USE_PREDICTION       = True   # False = Vorhersage komplett deaktivieren
PREDICTION_RECHECK   = 20    # Alle 20 Zyklen (5 Min) Vorhersage prüfen
USE_ML               = True  # True = KI-Modell statt regelbasierter Vorhersage
USE_PRICE_PREDICTOR  = True  # PricePredictor für Grid-Range (ersetzt calc_dynamic_range)

# ── Regime-abhängige Level-Anzahl ──────────────────────────────────────────────
# Range kommt IMMER vom PricePredictor (ATR/Bollinger-basiert, dynamisch).
# REGIME_CONFIGS steuert nur die Level-Anzahl – mehr Levels = mehr Fills = mehr Gewinn.
# Werte aus grid_optimizer.py – nach eigenem Lauf anpassen.
USE_REGIME_CONFIGS = True
REGIME_CONFIGS = {
    "ranging":  {"levels": 10},  # Seitwärts: mehr Levels = mehr Fills
    "trending": {"levels": 6},   # Trend: weniger Levels, breite Range → weniger Resets
    "volatile": {"levels": 14},  # Volatil: viele Levels = maximale Fills bei Bewegungen
}
# Grenzen für automatisches Tuning durch den Hintergrund-Optimizer
REGIME_LEVELS_MIN = {"ranging": 6, "trending": 4, "volatile": 10}
REGIME_LEVELS_MAX = {"ranging": 14, "trending": 8, "volatile": 18}

# ── Periodischer Grid-Rebuild ──────────────────────────────────────────────────
# Erzwingt Grid-Neuaufbau mit aktueller ATR-Range, auch wenn Preis noch in Range ist.
# 240 Zyklen × 15s = 60 Minuten. Setzt sicher dass PricePredictor immer aktuelle Daten nutzt.
GRID_REBUILD_CYCLES = 240

# ── Fee-Absicherung ────────────────────────────────────────────────────────────
# Kraken Maker-Fee ~0.16% pro Seite. Step muss 2× Fee überschreiten sonst Verlust.
KRAKEN_FEE = 0.0016
MIN_STEP_FEE_MULTIPLE = 2.5   # Step muss 2.5× Fee sein → Nettogewinn ≥ 0.48% pro Fill

# ── Adaptive Positionsgröße ────────────────────────────────────────────────────
# Bullisches Signal → tiefere Buy-Level bekommen mehr Budget (aggressives DCA)
# Bärisches Signal → gleichmäßiger oder reduziert
ADAPTIVE_SIZING   = True
SIZE_BIAS_FACTOR  = 0.30   # bis ±30% Abweichung vom gleichgroßen Trade

# ── Momentum-Hold beim Verkauf ─────────────────────────────────────────────────
# Wenn ML bullish + Momentum stark → Sell-Order ans nächste Level verschieben
# statt sofort zu verkaufen. Verhindert zu frühes Aussteigen in Trends.
MOMENTUM_HOLD_SCORE = 0.35  # direction_score > 0.35 → Hold und warte auf nächstes Level
MOMENTUM_HOLD_MAX   = 2     # Max. 2× verschieben pro Position (danach immer verkaufen)

_ml_predictor = None         # wird in run() initialisiert
_price_predictors: dict = {} # symbol → PricePredictor Instanz
_last_scores: dict = {}      # symbol → normierter Score (-1.0 … +1.0)


def _calc_level_allocations(grid_lines: list, current_price: float,
                             investment: float, direction_score: float) -> dict:
    """
    Verteilt das Investment nicht-gleichmäßig auf Grid-Level.
    Bullisch: tiefere Level (Käufe) bekommen mehr Budget → DCA nach unten.
    Bärisch:  höhere Level (Verkäufe) bekommen etwas mehr → früher sichern.
    """
    n = len(grid_lines)
    if not ADAPTIVE_SIZING or abs(direction_score) < 0.05 or n == 0:
        base = investment / n
        return {p: base for p in grid_lines}

    sorted_lines = sorted(grid_lines)
    bias = direction_score * SIZE_BIAS_FACTOR
    weights = []
    for i, price in enumerate(sorted_lines):
        rank = i / (n - 1) if n > 1 else 0.5   # 0 = tiefstes, 1 = höchstes Level
        if direction_score >= 0:
            # Bullisch: niedrige Level (rank≈0) bekommen Bonus
            w = 1.0 + bias * (0.5 - rank) * 2
        else:
            # Bärisch: hohe Level (rank≈1) bekommen Bonus
            w = 1.0 + abs(bias) * (rank - 0.5) * 2
        weights.append(max(0.2, w))

    total_w = sum(weights)
    return {p: w / total_w * investment for p, w in zip(sorted_lines, weights)}


def get_last_direction_score(symbol: str) -> float:
    """Letzter normierter Richtungs-Score für das Symbol (-1.0 … +1.0)."""
    return _last_scores.get(symbol, 0.0)


def predict_direction(symbol: str) -> str:
    """
    Gibt 'up', 'down' oder 'neutral' zurück.
    Verwendet KI-Modell wenn USE_ML=True, sonst regelbasierte Analyse.
    Speichert normierten Score in _last_scores für adaptive Positionsgrößen.
    """
    global _ml_predictor
    if USE_ML and _ml_predictor is not None:
        direction = _ml_predictor.predict(symbol)
        # Echten Blended-Score (LGBM+LLM) verwenden, nicht hardcoded ±0.5
        _last_scores[symbol] = _ml_predictor.get_score(symbol) or (
            0.5 if direction == "up" else (-0.5 if direction == "down" else 0.0)
        )
        return direction
    try:
        df = fetch_ohlcv(symbol, "1h", 100)
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        open_  = df["open"]
        volume = df["volume"]

        # Indikatoren
        ema9  = ta_lib.trend.ema_indicator(close, window=9).iloc[-1]
        ema21 = ta_lib.trend.ema_indicator(close, window=21).iloc[-1]
        rsi   = ta_lib.momentum.rsi(close, window=14).iloc[-1]
        mom   = (close.iloc[-1] - close.iloc[-4]) / close.iloc[-4] * 100

        macd_line   = ta_lib.trend.macd(close).iloc[-1]
        macd_signal = ta_lib.trend.macd_signal(close).iloc[-1]

        bb_high = ta_lib.volatility.bollinger_hband(close).iloc[-1]
        bb_low  = ta_lib.volatility.bollinger_lband(close).iloc[-1]
        price   = close.iloc[-1]
        bb_pct  = (price - bb_low) / (bb_high - bb_low) if bb_high != bb_low else 0.5

        vol_mean  = volume.rolling(20).mean().iloc[-1]
        vol_surge = volume.iloc[-1] > vol_mean * 1.5

        # Letzte 4 Kerzen für Multi-Candle-Muster
        c  = close.iloc[-1]; o  = open_.iloc[-1]; h  = high.iloc[-1]; l  = low.iloc[-1]
        c1 = close.iloc[-2]; o1 = open_.iloc[-2]; h1 = high.iloc[-2]; l1 = low.iloc[-2]
        c2 = close.iloc[-3]; o2 = open_.iloc[-3]
        body         = abs(c - o)
        total_range  = (h - l) if h != l else 1e-9
        lower_shadow = min(o, c) - l
        upper_shadow = h - max(o, c)

        score = 0

        # 1. EMA-Kreuzung
        if ema9 > ema21:  score += 1
        if ema9 < ema21:  score -= 1

        # 2. 3h-Momentum
        if mom > 0.5:   score += 1
        if mom < -0.5:  score -= 1

        # 3. RSI
        if rsi < 35:  score += 1
        if rsi > 65:  score -= 1

        # 4. MACD
        if macd_line > macd_signal:  score += 1
        if macd_line < macd_signal:  score -= 1

        # 5. Bollinger Band Position
        if bb_pct < 0.2:  score += 1
        if bb_pct > 0.8:  score -= 1

        # 6. Volumen-bestätigtes Momentum
        if vol_surge and mom > 0:  score += 1
        if vol_surge and mom < 0:  score -= 1

        # ── Einfache Candlestick-Muster ────────────────────────────────────────
        # 7. Hammer / Pin Bar bullisch
        if lower_shadow > 2 * body and upper_shadow < body and body / total_range < 0.4:
            score += 1

        # 8. Shooting Star bullisch
        if upper_shadow > 2 * body and lower_shadow < body and body / total_range < 0.4:
            score -= 1

        # 9. Bullish Engulfing
        if c1 < o1 and c > o and c >= o1 and o <= c1:
            score += 1

        # 10. Bearish Engulfing
        if c1 > o1 and c < o and c <= o1 and o >= c1:
            score -= 1

        # ── Mehr-Kerzen-Muster (Chart Patterns) ───────────────────────────────
        body1 = abs(c1 - o1); range1 = (h1 - l1) if h1 != l1 else 1e-9
        is_bull  = lambda _c, _o: _c > _o
        is_bear  = lambda _c, _o: _c < _o
        is_small = lambda b, r: b / r < 0.3

        # 11. Morning Star (bullische Umkehr): bärisch → klein → bullisch
        if is_bear(c2, o2) and is_small(body1, range1) and is_bull(c, o) and c > (o2 + c2) / 2:
            score += 2

        # 12. Evening Star (bärische Umkehr): bullisch → klein → bärisch
        if is_bull(c2, o2) and is_small(body1, range1) and is_bear(c, o) and c < (o2 + c2) / 2:
            score -= 2

        # 13. Three White Soldiers: drei bullische Kerzen, jede schließt höher
        if (is_bull(c, o) and is_bull(c1, o1) and is_bull(c2, o2)
                and c > c1 > c2 and o > o1 > o2):
            score += 2

        # 14. Three Black Crows: drei bärische Kerzen, jede schließt tiefer
        if (is_bear(c, o) and is_bear(c1, o1) and is_bear(c2, o2)
                and c < c1 < c2 and o < o1 < o2):
            score -= 2

        # 15. Doji (Unentschlossenheit): body < 5% der Range → neutralisiert letzten Tick
        if body / total_range < 0.05:
            score = int(score * 0.7)  # dämpfen, kein klares Signal

        # 16. RSI-Divergenz (vereinfacht): Preis neues Tief, RSI steigt → bullisch
        rsi_series = ta_lib.momentum.rsi(close, window=14)
        if (close.iloc[-1] < close.iloc[-5]
                and rsi_series.iloc[-1] > rsi_series.iloc[-5]):
            score += 1
        if (close.iloc[-1] > close.iloc[-5]
                and rsi_series.iloc[-1] < rsi_series.iloc[-5]):
            score -= 1

        direction = "up" if score >= 3 else "down" if score <= -3 else "neutral"
        _last_scores[symbol] = max(-1.0, min(1.0, score / 12.0))
        logger.info(
            "Vorhersage %-12s EMA=%+.2f RSI=%.1f Mom=%+.2f%% BB=%.2f Score=%+d → %s",
            symbol, ema9 - ema21, rsi, mom, bb_pct, score, direction.upper()
        )
        return direction
    except Exception as e:
        logger.warning("Vorhersage Fehler %s: %s", symbol, e)
        return "neutral"


def calc_dynamic_range(symbol: str) -> float:
    """Berechnet optimale Grid-Range basierend auf aktueller Volatilität (ATR)."""
    try:
        df = fetch_ohlcv(symbol, "1h", ATR_CANDLES + 5)
        atr = ta_lib.volatility.average_true_range(
            df["high"], df["low"], df["close"], window=ATR_CANDLES
        ).iloc[-1]
        price = df["close"].iloc[-1]
        volatility_pct = atr / price
        range_pct = min(max(volatility_pct * 3, GRID_RANGE_MIN), GRID_RANGE_MAX)
        logger.info("Volatilität %s: ATR=%.4f (%.1f%%) → Range ±%.1f%%",
                    symbol, atr, volatility_pct * 100, range_pct * 100)
        return range_pct
    except Exception:
        return 0.15  # Fallback


def _get_price_prediction(symbol: str) -> tuple:
    """
    Ruft PricePredictor ab → (lower, upper, regime, confidence).
    Bei Fehler: (None, None, '', 0.0) → Fallback auf ATR.
    """
    pp = _price_predictors.get(symbol)
    if pp is None:
        return None, None, "", 0.0
    try:
        result = pp.predict()
        low  = result["predicted_low"]
        high = result["predicted_high"]
        if low <= 0 or high <= low:
            raise ValueError(f"Ungültige Range: {low:.4f}–{high:.4f}")
        logger.info(
            "PricePredictor %s | Low=%.4f High=%.4f | Regime=%s Conf=%.2f",
            symbol, low, high, result["regime"], result["confidence"],
        )
        return low, high, result["regime"], result["confidence"]
    except Exception as e:
        logger.warning("PricePredictor Fehler %s: %s – Fallback auf ATR", symbol, e)
        return None, None, "", 0.0


def _build_grid_params(symbol: str, price: float, default_levels: int) -> tuple:
    """
    Berechnet optimale (lower, upper, levels, range_pct, regime, confidence).

    Logik für maximalen Tagesgewinn:
      1. Range vom PricePredictor (ATR/Bollinger – dynamisch)
      2. Level-Anzahl vom Regime-Config
      3. Min-Step wird erzwungen: Step ≥ MIN_STEP_FEE_MULTIPLE × 2 × KRAKEN_FEE
         Wenn PricePredictor-Range zu eng → Range wird aufgeweitet.
      4. Fallback auf ATR wenn PricePredictor fehlschlägt.
    """
    lower, upper, regime, confidence = _get_price_prediction(symbol)

    levels = default_levels
    if USE_REGIME_CONFIGS and regime in REGIME_CONFIGS:
        levels = REGIME_CONFIGS[regime]["levels"]

    # Min-Range basierend auf Fee (sonst Verlust pro Trade)
    min_range_pct = KRAKEN_FEE * levels * MIN_STEP_FEE_MULTIPLE

    if lower is not None and upper is not None:
        range_pct = (upper - lower) / (2 * price)
        if range_pct < min_range_pct:
            logger.info(
                "%s Range %.2f%% unter Min (%.2f%% für %d Levels) – aufgeweitet",
                symbol, range_pct * 100, min_range_pct * 100, levels,
            )
            lower = price * (1 - min_range_pct)
            upper = price * (1 + min_range_pct)
            range_pct = min_range_pct
    else:
        range_pct = max(calc_dynamic_range(symbol), min_range_pct)
        lower = price * (1 - range_pct)
        upper = price * (1 + range_pct)

    _log_expected_profit(symbol, price, levels, range_pct)
    return lower, upper, levels, range_pct, regime, confidence


def _log_expected_profit(symbol: str, price: float, levels: int, range_pct: float):
    """Loggt den erwarteten Gewinn pro Fill und pro Tag (Schätzung)."""
    step_pct    = range_pct * 2 / levels
    usdt_grid   = next((g["investment"] for g in GRIDS if g["symbol"] == symbol), 300.0)
    usdt_step   = usdt_grid / levels
    qty         = usdt_step / price
    profit_fill = step_pct * price * qty
    fee_fill    = 2 * price * qty * KRAKEN_FEE
    net_fill    = profit_fill - fee_fill
    # Grobe Schätzung: bei ~3% Daily-ATR und step_pct → fills_per_level_per_day ≈ 0.03/step_pct
    est_daily_atr_pct = 0.03
    fills_per_day = (est_daily_atr_pct / step_pct) * levels
    est_daily = net_fill * fills_per_day
    logger.info(
        "%s Grid | %d Levels | ±%.1f%% | Step=%.2f%% | "
        "Netto/Fill=%.4f USDT | ~%.2f USDT/Tag (bei 3%% ATR)",
        symbol, levels, range_pct * 100, step_pct * 100, net_fill, est_daily,
    )

class PaperGridBot:
    def __init__(self, symbol: str, investment: float, levels: int, range_pct: float):
        self.symbol = symbol
        self.investment = investment
        self._initial_investment = investment  # Cap für Compounding
        self.levels = levels
        self.range_pct = range_pct

        self.grid_lines: list[float] = []
        self.orders: dict[float, dict] = {}
        self.total_profit = 0.0
        self.trade_count = 0
        self.usdt_per_grid = investment / levels
        self._last_compound_at = 0
        self._compounded_profit = 0.0   # bereits reinvestierter Profit
        self._direction_score = 0.0     # für adaptive Positionsgröße
        self._level_allocations: dict = {}
        self.with_position = True
        self._last_regime = ""
        self._last_confidence = 0.0
        self._last_pred_low = 0.0
        self._last_pred_high = 0.0
        self._directional: dict = {}  # aktiver Directional Trade: {qty, entry, tp, sl, usdt}

    def _maybe_open_directional(self, current_price: float):
        """Öffnet einen Directional Trade wenn ML 'up' signalisiert und kein Trade offen."""
        if not DIRECTIONAL_ENABLED:
            return
        if self._directional:
            return  # bereits offen
        # Kaufen wenn Direction UP ist (Dashboard zeigt UP → Bot kauft)
        if self._last_prediction != "up":
            return
        if self._direction_score < DIRECTIONAL_SCORE_MIN:
            return

        # ATR aus letzten 24 Candles schätzen (vereinfacht: range_pct als Proxy)
        atr = current_price * self.range_pct * 0.5
        usdt = self.investment * DIRECTIONAL_PCT
        qty  = usdt / current_price
        tp   = current_price + DIRECTIONAL_TP_ATR * atr
        sl   = current_price - DIRECTIONAL_SL_ATR * atr

        self._directional = {
            "entry": current_price, "qty": qty,
            "usdt": usdt, "tp": tp, "sl": sl,
        }
        logger.info(
            "[DIRECTIONAL] %s KAUF @ %.4f | %.2f USDT | TP=%.4f (+%.1f%%) | SL=%.4f (-%.1f%%)",
            self.symbol, current_price, usdt,
            tp, (tp / current_price - 1) * 100,
            sl, (1 - sl / current_price) * 100,
        )

    def _check_directional(self, current_price: float):
        """Prüft ob TP oder SL des Directional Trades erreicht wurde."""
        if not self._directional:
            return
        d = self._directional
        hit_tp = current_price >= d["tp"]
        hit_sl = current_price <= d["sl"]
        score  = self._direction_score

        # Auch schließen wenn Signal dreht
        signal_flipped = score < 0

        if not (hit_tp or hit_sl or signal_flipped):
            return

        pnl = (current_price - d["entry"]) * d["qty"]
        fee = (current_price + d["entry"]) * d["qty"] * KRAKEN_FEE
        net = pnl - fee
        self.total_profit += net
        self.trade_count  += 1

        reason = "TP" if hit_tp else ("SL" if hit_sl else "Signal-Flip")
        logger.info(
            "[DIRECTIONAL] %s VERKAUF @ %.4f | Grund: %s | PnL: %+.4f USDT",
            self.symbol, current_price, reason, net,
        )
        try:
            from dashboard.db import log_trade
            log_trade(self.symbol, "DIRECTIONAL", d["entry"], current_price,
                      net, f"directional_{reason.lower()}", "grid",
                      "paper" if config.PAPER_TRADING else "live")
        except Exception:
            pass
        self._directional = {}

    def setup_grid(self, current_price: float,
                   lower: float = None, upper: float = None):
        if lower is None:
            lower = current_price * (1 - self.range_pct)
        if upper is None:
            upper = current_price * (1 + self.range_pct)
        step = (upper - lower) / self.levels

        # Half-step offset: current_price liegt immer ZWISCHEN zwei Levels, nie genau drauf.
        self.grid_lines = [lower + (i + 0.5) * step for i in range(self.levels)]
        self.orders = {}

        # Adaptive Positionsgröße: mehr Budget für tiefere Level wenn bullisch
        self._level_allocations = _calc_level_allocations(
            self.grid_lines, current_price, self.investment, self._direction_score
        )
        self.usdt_per_grid = self.investment / self.levels  # Durchschnitt für Logging

        for i, price in enumerate(self.grid_lines):
            usdt = self._level_allocations.get(price, self.usdt_per_grid)
            qty = usdt / price
            if price < current_price:
                self.orders[price] = {"side": "buy", "qty": qty, "filled": False}
            elif self.with_position:
                buy_price = self.grid_lines[i - 1] if i > 0 else current_price
                buy_usdt = self._level_allocations.get(buy_price, self.usdt_per_grid)
                self.orders[price] = {
                    "side": "sell", "qty": buy_usdt / buy_price,
                    "filled": False, "bought_at": buy_price,
                }
            else:
                self.orders[price] = {"side": "sell", "qty": qty, "filled": False}

        logger.info(
            "Grid aufgebaut | %s | %.4f – %.4f | %d Stufen | ø %.2f USDT/Stufe | Score=%+.2f",
            self.symbol, lower, upper, self.levels, self.usdt_per_grid, self._direction_score
        )
        self._print_grid(current_price)

    def _print_grid(self, current_price: float):
        print("\n" + "="*45)
        print(f"  GRID – {self.symbol}")
        print("="*45)
        price_printed = False
        for price in sorted(self.grid_lines, reverse=True):
            if not price_printed and price < current_price:
                print(f"  {current_price:>10.4f} USDT  ◄── AKTUELL")
                price_printed = True
            order = self.orders[price]
            filled = " [FILLED]" if order["filled"] else ""
            print(f"  {price:>10.4f} USDT  {order['side'].upper():<5}{filled}")
        if not price_printed:
            print(f"  {current_price:>10.4f} USDT  ◄── AKTUELL")
        print("="*45 + "\n")

    def _check_position_stop_losses(self, current_price: float):
        """Schließt Positionen deren Per-Position-Stop-Loss getroffen wurde."""
        for price, order in list(self.orders.items()):
            if order.get("filled") or order["side"] != "sell":
                continue
            if "bought_at" not in order or "sl_price" not in order:
                continue
            if current_price <= order["sl_price"]:
                buy_price = order["bought_at"]
                qty = order["qty"]
                profit = (current_price - buy_price) * qty
                fee = (current_price + buy_price) * qty * KRAKEN_FEE
                net_profit = profit - fee
                self.total_profit += net_profit
                self.trade_count += 1
                order["filled"] = True
                logger.warning(
                    "[STOP-LOSS] %s @ %.4f | Gekauft @ %.4f | SL @ %.4f | Verlust: %.4f USDT",
                    self.symbol, current_price, buy_price, order["sl_price"], net_profit,
                )
                try:
                    from dashboard.db import log_trade
                    log_trade(self.symbol, "SELL", buy_price, current_price, net_profit,
                              "stop_loss", "grid",
                              "paper" if config.PAPER_TRADING else "live")
                except Exception:
                    pass

    def check_fills(self, current_price: float):
        self._check_position_stop_losses(current_price)
        self._check_directional(current_price)
        self._maybe_open_directional(current_price)

        for price, order in list(self.orders.items()):
            if order["filled"]:
                continue

            # Kauf-Order: Preis fällt auf oder unter Grid-Level
            if order["side"] == "buy" and current_price <= price:
                if not self.with_position:
                    # ML sagt DOWN → kein neuer Kauf, Order überspringen
                    continue
                order["filled"] = True
                order["fill_price"] = price
                idx = self.grid_lines.index(price)
                if idx < len(self.grid_lines) - 1:
                    sell_price = self.grid_lines[idx + 1]
                    # SL = 2× Stepgröße unter Kaufpreis (proportional, nicht fix 4%)
                    step_pct = (sell_price - price) / price
                    sl_pct   = max(step_pct * 2, PER_POS_SL_PCT)
                    sl_price = price * (1 - sl_pct)
                    self.orders[sell_price] = {
                        "side": "sell",
                        "qty": order["qty"],
                        "filled": False,
                        "bought_at": price,
                        "sl_price": sl_price,
                    }
                    logger.info("[PAPER GRID] BUY  %s @ %.4f | Qty: %.4f | Wert: %.2f USDT | SL @ %.4f (%.1f%%)",
                                self.symbol, price, order["qty"], price * order["qty"],
                                sl_price, sl_pct * 100)
                else:
                    logger.info("[PAPER GRID] BUY  %s @ %.4f | Qty: %.4f | Wert: %.2f USDT | oberstes Level",
                                self.symbol, price, order["qty"], price * order["qty"])

            # Verkauf-Order: nur wenn vorher wirklich gekauft wurde
            elif order["side"] == "sell" and current_price >= price and "bought_at" in order:
                # Momentum-Hold: bei bullischem Signal Sell ans nächste Level schieben
                direction_score = get_last_direction_score(self.symbol)
                holds = order.get("momentum_holds", 0)
                if direction_score > MOMENTUM_HOLD_SCORE and holds < MOMENTUM_HOLD_MAX:
                    try:
                        idx = self.grid_lines.index(price)
                        if idx < len(self.grid_lines) - 1:
                            next_price = self.grid_lines[idx + 1]
                            order["momentum_holds"] = holds + 1
                            del self.orders[price]
                            self.orders[next_price] = order
                            logger.info(
                                "[MOMENTUM HOLD] %s Sell %.4f→%.4f (Score=%.2f, %d/%d)",
                                self.symbol, price, next_price,
                                direction_score, holds + 1, MOMENTUM_HOLD_MAX,
                            )
                            continue
                    except (ValueError, IndexError):
                        pass  # Level nicht gefunden → normal verkaufen

                order["filled"] = True
                buy_price = order["bought_at"]
                profit = (price - buy_price) * order["qty"]
                fee = (price + buy_price) * order["qty"] * KRAKEN_FEE
                net_profit = profit - fee
                self.total_profit += net_profit
                self.trade_count += 1
                hold_note = f" (nach {holds}× Hold)" if holds else ""
                logger.info("[PAPER GRID] SELL %s @ %.4f | Gekauft @ %.4f | Profit: %.4f USDT%s",
                            self.symbol, price, buy_price, net_profit, hold_note)
                notifier.notify_trade_close(
                    self.symbol, "GRID", buy_price, price, net_profit, "grid_fill"
                )
                try:
                    from dashboard.db import log_trade
                    log_trade(self.symbol, "GRID", buy_price, price, net_profit,
                              "grid_fill", "grid",
                              "paper" if config.PAPER_TRADING else "live")
                except Exception:
                    pass
                # Neue Kauf-Order am alten Kauflevel – nur wenn ML nicht DOWN
                if self.with_position:
                    replenish_usdt = self._level_allocations.get(buy_price, self.usdt_per_grid)
                    self.orders[buy_price] = {
                        "side": "buy",
                        "qty": replenish_usdt / buy_price,
                        "filled": False,
                    }
                # Auto-Compounding: Gewinne alle X Trades reinvestieren
                self._maybe_compound(price)

    def _maybe_compound(self, current_price: float):
        trades_since_last = self.trade_count - self._last_compound_at
        if self.total_profit <= 0 or trades_since_last < COMPOUND_EVERY_TRADES:
            return

        profit_delta = self.total_profit - self._compounded_profit
        if profit_delta <= 0:
            return

        old_investment = self.investment
        max_investment = self._initial_investment * MAX_INVESTMENT_MULT
        self.investment = min(self.investment + profit_delta, max_investment)
        actual_delta = self.investment - old_investment
        self.usdt_per_grid = self.investment / self.levels
        self._last_compound_at = self.trade_count
        self._compounded_profit = self.total_profit

        cap_note = f" (Cap: {max_investment:.0f} USDT)" if self.investment >= max_investment else ""
        logger.info("♻️  COMPOUND %s | %.2f → %.2f USDT (+%.2f)%s | Neu: %.2f/Grid",
                    self.symbol, old_investment, self.investment,
                    actual_delta, cap_note, self.usdt_per_grid)
        notifier._send(
            f"♻️ <b>Auto-Compound</b> {self.symbol}\n"
            f"Investment: {old_investment:.2f} → {self.investment:.2f} USDT\n"
            f"Gewinn reinvestiert: +{actual_delta:.2f} USDT{cap_note}"
        )
        self.setup_grid(current_price)

    def emergency_sell(self, current_price: float, reason: str = "Vorhersage DOWN"):
        """Verkauft alle offenen Positionen sofort zum aktuellen Preis."""
        sold_count = 0
        for _, order in self.orders.items():
            if order["side"] == "sell" and not order["filled"] and "bought_at" in order:
                buy_price = order["bought_at"]
                qty = order["qty"]
                profit = (current_price - buy_price) * qty
                fee = (current_price + buy_price) * qty * KRAKEN_FEE
                net_profit = profit - fee
                self.total_profit += net_profit
                self.trade_count += 1
                order["filled"] = True
                sold_count += 1
                logger.info("[NOTVERKAUF] %s @ %.4f | Gekauft @ %.4f | Profit: %.4f USDT",
                            self.symbol, current_price, buy_price, net_profit)
                try:
                    from dashboard.db import log_trade
                    log_trade(self.symbol, "SELL", buy_price, current_price, net_profit,
                              reason, "grid", "paper" if config.PAPER_TRADING else "live")
                except Exception:
                    pass
        if sold_count:
            logger.warning("[NOTVERKAUF] %s – %d Positionen geschlossen | Grund: %s",
                           self.symbol, sold_count, reason)
            notifier._send(
                f"🔴 <b>Notverkauf {self.symbol}</b>\n"
                f"{sold_count} Positionen @ {current_price:.4f}\nGrund: {reason}"
            )

    def check_stop_loss(self) -> bool:
        """Notbremse – gibt True zurück wenn Bot stoppen soll. Limit skaliert mit Investment."""
        max_loss = self.investment * MAX_LOSS_PCT
        if self.total_profit <= -max_loss:
            logger.warning(
                "NOTBREMSE %s: Verlust %.2f USDT ≥ %.0f%% von %.0f USDT Investment – Bot stoppt!",
                self.symbol, self.total_profit, MAX_LOSS_PCT * 100, self.investment,
            )
            notifier._send(
                f"🚨 <b>NOTBREMSE ausgelöst!</b>\n{self.symbol}\n"
                f"Gesamtverlust: {self.total_profit:.2f} USDT (Limit: -{max_loss:.0f} USDT)\n"
                f"Bot wurde gestoppt."
            )
            return True
        return False

    def status(self) -> str:
        filled = sum(1 for o in self.orders.values() if o["filled"])
        return (f"Trades: {self.trade_count} | "
                f"Gefüllte Orders: {filled}/{len(self.orders)} | "
                f"Profit: {self.total_profit:+.4f} USDT")


class LiveGridBot:
    """Echter Grid Bot – platziert Limit-Orders auf Kraken."""

    def __init__(self, symbol: str, investment: float, levels: int, range_pct: float):
        self.symbol     = symbol
        self.investment = investment
        self._initial_investment = investment  # Cap für Compounding
        self.levels     = levels
        self.range_pct  = range_pct

        self.grid_lines: list[float]  = []
        self.open_orders: dict[str, dict] = {}  # order_id → {side, price, qty, bought_at}
        self.total_profit = 0.0
        self.trade_count  = 0
        self.usdt_per_grid = investment / levels
        self._last_compound_at = 0
        self._compounded_profit = 0.0
        self._direction_score = 0.0
        self._level_allocations: dict = {}
        self.with_position = True
        self._last_regime = ""
        self._last_confidence = 0.0
        self._last_pred_low = 0.0
        self._last_pred_high = 0.0

    def _exchange(self):
        from data_fetcher import get_exchange
        return get_exchange()

    def _cancel_all(self):
        ex = self._exchange()
        for oid in list(self.open_orders.keys()):
            try:
                ex.cancel_order(oid, self.symbol)
                logger.info("Order storniert: %s", oid)
            except Exception as e:
                logger.warning("Storno-Fehler %s: %s", oid, e)
        self.open_orders = {}

    def setup_grid(self, current_price: float,
                   lower: float = None, upper: float = None):
        self._cancel_all()
        ex = self._exchange()

        if lower is None:
            lower = current_price * (1 - self.range_pct)
        if upper is None:
            upper = current_price * (1 + self.range_pct)
        step  = (upper - lower) / self.levels
        self.grid_lines = [lower + (i + 0.5) * step for i in range(self.levels)]
        self.usdt_per_grid = self.investment / self.levels

        # Adaptive Positionsgröße
        self._level_allocations = _calc_level_allocations(
            self.grid_lines, current_price, self.investment, self._direction_score
        )

        sell_levels = [p for p in self.grid_lines if p > current_price]
        buy_levels  = [p for p in self.grid_lines if p < current_price]

        # Coins für obere Hälfte kaufen – nur wenn Vorhersage "up"/"neutral"
        if sell_levels and self.with_position:
            total_usdt = sum(self._level_allocations.get(p, self.usdt_per_grid) for p in sell_levels)
            total_qty = round(total_usdt / current_price, 6)
            try:
                ex.create_market_order(self.symbol, "buy", total_qty)
                logger.info("[LIVE] Markt-Kauf (Vorhersage UP): %.6f @ ~%.4f USDT",
                            total_qty, current_price)
                time.sleep(1)
            except Exception as e:
                logger.error("Markt-Kauf Fehler: %s", e)

        # Limit-Buy-Orders unterhalb des Preises
        for price in buy_levels:
            usdt = self._level_allocations.get(price, self.usdt_per_grid)
            qty = round(usdt / price, 6)
            try:
                order = ex.create_limit_order(self.symbol, "buy", qty, price)
                self.open_orders[order["id"]] = {
                    "side": "buy", "price": price, "qty": qty, "bought_at": None
                }
                logger.info("[LIVE] BUY Order @ %.4f | Qty: %.6f | ID: %s",
                            price, qty, order["id"])
                time.sleep(0.3)
            except Exception as e:
                logger.error("BUY Order-Fehler @ %.4f: %s", price, e)

        # Limit-Sell-Orders oberhalb (Coins bereits gekauft)
        for i, price in enumerate(self.grid_lines):
            if price <= current_price:
                continue
            buy_price = self.grid_lines[i - 1] if i > 0 else current_price
            buy_usdt = self._level_allocations.get(buy_price, self.usdt_per_grid)
            qty = round(buy_usdt / buy_price, 6)
            try:
                order = ex.create_limit_order(self.symbol, "sell", qty, price)
                self.open_orders[order["id"]] = {
                    "side": "sell", "price": price, "qty": qty, "bought_at": buy_price
                }
                logger.info("[LIVE] SELL Order @ %.4f | Qty: %.6f | ID: %s",
                            price, qty, order["id"])
                time.sleep(0.3)
            except Exception as e:
                logger.error("SELL Order-Fehler @ %.4f: %s", price, e)

        logger.info("Grid aufgebaut | %s | %.4f – %.4f | %d Buy / %d Sell Orders",
                    self.symbol, lower, upper, len(buy_levels), len(sell_levels))

    def check_fills(self, current_price: float):
        ex = self._exchange()
        try:
            closed = ex.fetch_closed_orders(self.symbol, limit=20)
        except Exception as e:
            logger.warning("Fetch orders Fehler: %s", e)
            return

        closed_ids = {o["id"]: o for o in closed if o["status"] == "closed"}

        for oid, info in list(self.open_orders.items()):
            if oid not in closed_ids:
                continue

            filled_order = closed_ids[oid]
            fill_price   = filled_order.get("average") or info["price"]
            qty          = filled_order.get("filled") or info["qty"]

            if info["side"] == "buy":
                logger.info("[LIVE] BUY gefüllt @ %.4f | Qty: %.6f", fill_price, qty)
                idx = self.grid_lines.index(
                    min(self.grid_lines, key=lambda x: abs(x - info["price"]))
                )
                # Momentum-Hold: bei bullischem Signal 2 Levels höher verkaufen statt 1
                direction_score = get_last_direction_score(self.symbol)
                step_up = 2 if direction_score > MOMENTUM_HOLD_SCORE and idx < len(self.grid_lines) - 2 else 1
                sell_idx = min(idx + step_up, len(self.grid_lines) - 1)
                sell_price = self.grid_lines[sell_idx]
                if step_up == 2:
                    logger.info("[MOMENTUM TARGET] %s Sell-Ziel auf %.4f (+2 Levels, Score=%.2f)",
                                self.symbol, sell_price, direction_score)
                try:
                    sell_order = ex.create_limit_order(self.symbol, "sell", qty, sell_price)
                    self.open_orders[sell_order["id"]] = {
                        "side": "sell", "price": sell_price,
                        "qty": qty, "bought_at": fill_price
                    }
                    logger.info("[LIVE] SELL Order @ %.4f | ID: %s", sell_price, sell_order["id"])
                except Exception as e:
                    logger.error("SELL Order Fehler: %s", e)
                del self.open_orders[oid]

            elif info["side"] == "sell" and info["bought_at"]:
                buy_price  = info["bought_at"]
                profit     = (fill_price - buy_price) * qty
                fee        = (fill_price + buy_price) * qty * KRAKEN_FEE
                net_profit = profit - fee
                self.total_profit += net_profit
                self.trade_count  += 1
                logger.info("[LIVE] SELL gefüllt @ %.4f | Profit: %.4f USDT", fill_price, net_profit)
                notifier.notify_trade_close(
                    self.symbol, "GRID", buy_price, fill_price, net_profit, "grid_fill"
                )
                # Kauf-Order zurück auf altes Level
                try:
                    new_order = ex.create_limit_order(self.symbol, "buy", qty, buy_price)
                    self.open_orders[new_order["id"]] = {
                        "side": "buy", "price": buy_price, "qty": qty, "bought_at": None
                    }
                except Exception as e:
                    logger.error("Replenish BUY Fehler: %s", e)
                del self.open_orders[oid]
                self._maybe_compound(current_price)

    def _maybe_compound(self, current_price: float):
        trades_since = self.trade_count - self._last_compound_at
        if self.total_profit <= 0 or trades_since < COMPOUND_EVERY_TRADES:
            return
        profit_delta = self.total_profit - self._compounded_profit
        if profit_delta <= 0:
            return
        old = self.investment
        max_investment = self._initial_investment * MAX_INVESTMENT_MULT
        self.investment = min(self.investment + profit_delta, max_investment)
        actual_delta = self.investment - old
        self.usdt_per_grid = self.investment / self.levels
        self._last_compound_at = self.trade_count
        self._compounded_profit = self.total_profit
        cap_note = f" (Cap: {max_investment:.0f} USDT)" if self.investment >= max_investment else ""
        logger.info("♻️  COMPOUND %s | %.2f → %.2f USDT (+%.2f)%s", self.symbol, old, self.investment, actual_delta, cap_note)
        notifier._send(f"♻️ <b>Compound</b> {self.symbol}\n{old:.2f} → {self.investment:.2f} USDT (+{actual_delta:.2f}){cap_note}")
        self.setup_grid(current_price)

    def check_stop_loss(self) -> bool:
        max_loss = self.investment * MAX_LOSS_PCT
        if self.total_profit <= -max_loss:
            logger.warning(
                "NOTBREMSE %s: Verlust %.2f USDT ≥ %.0f%% von %.0f USDT Investment",
                self.symbol, self.total_profit, MAX_LOSS_PCT * 100, self.investment,
            )
            notifier._send(
                f"🚨 <b>NOTBREMSE</b> {self.symbol}\n"
                f"Verlust: {self.total_profit:.2f} USDT (Limit: -{max_loss:.0f} USDT)"
            )
            self._cancel_all()
            return True
        return False

    @property
    def orders(self) -> dict:
        """Orders im PaperGridBot-Format für Dashboard-Kompatibilität."""
        return {
            info["price"]: {
                "side":      info["side"],
                "filled":    False,
                "bought_at": info.get("bought_at"),
            }
            for info in self.open_orders.values()
            if "price" in info
        }

    def status(self) -> str:
        return (f"Trades: {self.trade_count} | "
                f"Offene Orders: {len(self.open_orders)} | "
                f"Profit: {self.total_profit:+.4f} USDT")


AUTO_TUNE_INTERVAL_H = 6     # Hintergrund-Optimizer alle 6 Stunden
AUTO_TUNE_MIN_TRADES = 20    # Mind. X Trades pro Regime nötig für Anpassung
AUTO_TUNE_WIN_THRESH = 0.55  # Win-Rate > 55% → +1 Level; < 45% → -1 Level


def _auto_tune_once():
    """Liest Trade-Historie, passt REGIME_CONFIGS-Levels nach echten Win-Rates an."""
    try:
        from dashboard.db import get_conn
        con = get_conn()
        rows = con.execute("""
            SELECT tc.regime, COUNT(*) AS n,
                   SUM(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END) AS wins
            FROM trades t
            JOIN trade_context tc ON tc.trade_id = t.id
            WHERE t.timestamp >= datetime('now', '-7 days')
              AND tc.regime IS NOT NULL
            GROUP BY tc.regime
        """).fetchall()
        con.close()
    except Exception as e:
        logger.debug("Auto-Tune: DB-Fehler %s", e)
        return

    if not rows:
        logger.debug("Auto-Tune: noch keine trade_context-Daten – übersprungen.")
        return

    changed = []
    for row in rows:
        regime = row["regime"]
        n      = row["n"]
        wins   = row["wins"]
        if regime not in REGIME_CONFIGS or n < AUTO_TUNE_MIN_TRADES:
            continue
        win_rate   = wins / n
        current    = REGIME_CONFIGS[regime]["levels"]
        lo         = REGIME_LEVELS_MIN.get(regime, 4)
        hi         = REGIME_LEVELS_MAX.get(regime, 18)
        if win_rate > AUTO_TUNE_WIN_THRESH and current < hi:
            REGIME_CONFIGS[regime]["levels"] = current + 1
            changed.append(f"{regime}: {current}→{current+1} (WR={win_rate:.0%}, n={n})")
        elif win_rate < (1 - AUTO_TUNE_WIN_THRESH) and current > lo:
            REGIME_CONFIGS[regime]["levels"] = current - 1
            changed.append(f"{regime}: {current}→{current-1} (WR={win_rate:.0%}, n={n})")

    if changed:
        logger.info("Auto-Tune REGIME_CONFIGS: %s", " | ".join(changed))
        notifier._send("🔧 <b>Auto-Tune</b>\n" + "\n".join(changed))
    else:
        logger.debug("Auto-Tune: Keine Anpassung nötig.")


def _start_auto_tuner():
    def _loop():
        time.sleep(AUTO_TUNE_INTERVAL_H * 3600)
        while _running:
            _auto_tune_once()
            time.sleep(AUTO_TUNE_INTERVAL_H * 3600)
    t = threading.Thread(target=_loop, name="auto-tuner", daemon=True)
    t.start()
    logger.info("Hintergrund-Optimizer gestartet (alle %dh).", AUTO_TUNE_INTERVAL_H)


def run():
    # Coin-Budget aus Dashboard-Settings lesen (überschreibt GRIDS-Defaults)
    try:
        from dashboard.db import init_coin_settings, get_all_coin_settings
        init_coin_settings([(g["symbol"], g["investment"]) for g in GRIDS])
        db_settings = {s["symbol"]: s for s in get_all_coin_settings()}
        for g in GRIDS:
            sym = g["symbol"]
            if sym in db_settings:
                new_inv = db_settings[sym]["max_investment"]
                if new_inv != g["investment"]:
                    logger.info("Budget %s: %.0f → %.0f USDT (Dashboard-Setting)",
                                sym, g["investment"], new_inv)
                g["investment"] = new_inv
                g["enabled"] = db_settings[sym].get("enabled", 1)
    except Exception as e:
        logger.warning("Coin-Settings nicht geladen: %s", e)

    active_grids = [g for g in GRIDS if g.get("enabled", 1)]
    total_investment = sum(g["investment"] for g in active_grids)
    symbols = [g["symbol"] for g in active_grids]
    mode = "PAPER" if config.PAPER_TRADING else "LIVE"
    BotClass = PaperGridBot if config.PAPER_TRADING else LiveGridBot

    # ── Cross-Coin Risk-Manager (Daily-Drawdown -3%) ───────────────────────────
    from src.risk.risk_manager import RiskManager
    _risk_manager = RiskManager(
        params={"max_daily_drawdown": 0.03, "max_open_positions": 99, "max_portfolio_risk": 1.0},
        initial_capital=total_investment,
    )
    _freeze_mode = False  # wenn True: keine neuen Buy-Orders, nur Sells

    logger.info("="*55)
    logger.info("Multi-Grid Bot startet | %s | %d Coins | %.0f USDT total",
                mode, len(GRIDS), total_investment)
    logger.info("="*55)

    _start_auto_tuner()

    if not config.PAPER_TRADING:
        logger.warning("⚠️  LIVE TRADING AKTIV – echtes Geld wird gehandelt!")

    # ── KI-Modell initialisieren ───────────────────────────────────────────────
    global _ml_predictor, _price_predictors
    if USE_ML and USE_PREDICTION:
        from ml import MLPredictor
        from data_fetcher import fetch_ohlcv
        _ml_predictor = MLPredictor(fetch_ohlcv)
        logger.info("KI-Modell wird initialisiert (Bootstrap aus 1000 Candles je Coin)…")
        _ml_predictor.initialize(symbols)
        logger.info("KI-Modell bereit.")

    # ── PricePredictor initialisieren ──────────────────────────────────────────
    if USE_PRICE_PREDICTOR:
        from price_predictor import PricePredictor
        for g in active_grids:
            sym = g["symbol"]
            _price_predictors[sym] = PricePredictor(
                exchange_id=config.EXCHANGE_ID,
                symbol=sym,
                timeframe="1h",
                limit=200,
                grid_count=g["levels"],
            )
        logger.info("PricePredictor initialisiert für %d Coins", len(_price_predictors))

    def _log_bot_state(bot, price: float):
        """Schreibt Grid-State sofort in DB – verhindert veraltete filled-Häkchen nach Restart."""
        try:
            from dashboard.db import update_grid_state
            update_grid_state(
                bot.symbol, price, bot.orders,
                bot.range_pct, bot.investment,
                bot.total_profit, bot.trade_count,
                getattr(bot, "_last_prediction", ""),
                predicted_low=bot._last_pred_low,
                predicted_high=bot._last_pred_high,
                confidence=bot._last_confidence,
                regime=bot._last_regime,
            )
        except Exception:
            pass

    def _apply_prediction(bot) -> bool:
        """
        Gibt True zurück wenn sich with_position geändert hat.
        Grid Bot hält Positionen bei UP und NEUTRAL – nur bei DOWN wird alles verkauft.
        Setzt auch bot._direction_score für adaptive Positionsgrößen.
        """
        if USE_PREDICTION:
            direction = predict_direction(bot.symbol)
            bot._direction_score = get_last_direction_score(bot.symbol)
            new_pos = direction != "down"
            changed = new_pos != bot.with_position
            bot.with_position = new_pos
            bot._last_prediction = direction
            if changed:
                notifier._send(
                    f"🔮 <b>Vorhersage {bot.symbol}</b>: {direction.upper()}\n"
                    f"{'➡ Position aktiv' if bot.with_position else '⬇ Abwärtstrend – kein Einstieg'}"
                )
            return changed
        else:
            bot.with_position = True
            bot._last_prediction = "neutral"
            bot._direction_score = 0.0
            return False

    # Grids initialisieren mit PricePredictor-Range + ML-Vorhersage
    bots = []
    for g in active_grids:
        sym = g["symbol"]
        try:
            ticker = fetch_ticker(sym)
            price  = ticker["last"]
        except Exception as e:
            logger.error("Startup: Ticker für %s nicht verfügbar (%s) – übersprungen", sym, e)
            continue

        try:
            lower, upper, levels, range_pct, regime, confidence = _build_grid_params(
                sym, price, g["levels"]
            )
            bot = BotClass(sym, g["investment"], levels, range_pct)
            bot._last_regime     = regime
            bot._last_confidence = confidence
            bot._last_pred_low   = lower
            bot._last_pred_high  = upper

            _apply_prediction(bot)
            bot.setup_grid(price, lower=lower, upper=upper)
            _log_bot_state(bot, price)
            bots.append(bot)
        except Exception as e:
            logger.error("Startup: Grid-Init für %s fehlgeschlagen: %s", sym, e)
        time.sleep(0.5)

    range_src = "PricePredictor" if USE_PRICE_PREDICTOR else "ATR"
    notifier._send(
        f"🔲 <b>Multi-Grid Bot gestartet ({mode})</b>\n"
        f"Coins: {', '.join(symbols)}\n"
        f"Budget: {total_investment:.0f} USDT total\n"
        f"Vorhersage: {'AN' if USE_PREDICTION else 'AUS'} | ML: {'AN' if USE_ML else 'AUS'} | Range: {range_src}"
    )

    loop_count = 0
    while _running:
        loop_count += 1
        try:
            prices: dict[str, float] = {}
            for bot in bots:
                try:
                    prices[bot.symbol] = fetch_ticker(bot.symbol)["last"]
                    time.sleep(0.5)  # Rate Limit respektieren
                except Exception as e:
                    logger.warning("Ticker Fehler %s: %s", bot.symbol, e)
                    continue

            # Cross-Coin Daily-Drawdown prüfen
            total_capital_now = sum(b.investment for b in bots)
            total_profit_now  = sum(b.total_profit for b in bots)
            if not _risk_manager.check_daily_drawdown(total_capital_now + total_profit_now):
                if not _freeze_mode:
                    _freeze_mode = True
                    logger.warning(
                        "FREEZE-MODE: Tages-Drawdown ≥3%% – keine neuen Buy-Orders bis morgen!"
                    )
                    notifier._send(
                        "❄️ <b>Freeze-Mode aktiviert</b>\n"
                        "Tages-Drawdown ≥3%% – Bot kauft nicht mehr bis morgen.\n"
                        f"Gesamt-Profit heute: {total_profit_now:+.2f} USDT"
                    )
            else:
                if _freeze_mode:
                    _freeze_mode = False
                    logger.info("Freeze-Mode aufgehoben – neuer Handelstag.")

            for bot in bots:
                current_price = prices.get(bot.symbol)
                if current_price is None:
                    continue

                if bot.check_stop_loss():
                    logger.warning("%s Grid gestoppt (Notbremse)", bot.symbol)
                    continue

                if _freeze_mode:
                    bot.check_fills(current_price)  # nur Sells abwickeln
                    continue

                bot.check_fills(current_price)

                grid_lo = bot.grid_lines[0]
                grid_hi = bot.grid_lines[-1]
                out_of_range  = current_price < grid_lo * 0.99 or current_price > grid_hi * 1.01
                do_recheck    = USE_PREDICTION and loop_count % PREDICTION_RECHECK == 0
                do_rebuild    = loop_count % GRID_REBUILD_CYCLES == 0  # stündlicher Zwangs-Rebuild

                if out_of_range or do_recheck or do_rebuild:
                    lower, upper, levels, new_range, regime, confidence = _build_grid_params(
                        bot.symbol, current_price, bot.levels
                    )
                    if levels != bot.levels:
                        logger.info("%s Levels %d→%d (Regime: %s)",
                                    bot.symbol, bot.levels, levels, regime)
                        bot.levels = levels
                        bot.usdt_per_grid = bot.investment / bot.levels

                    bot.range_pct        = new_range
                    bot._last_regime     = regime
                    bot._last_confidence = confidence
                    bot._last_pred_low   = lower
                    bot._last_pred_high  = upper

                    prediction_flipped = _apply_prediction(bot)
                    if out_of_range or prediction_flipped or do_rebuild:
                        if out_of_range:
                            reason = "außerhalb Range"
                        elif prediction_flipped:
                            reason = "Vorhersage geändert"
                        else:
                            reason = "stündlicher Rebuild"
                        logger.warning(
                            "%s %s – Grid neu (±%.1f%%, %d Levels, Regime=%s, Conf=%.2f)",
                            bot.symbol, reason, new_range * 100, bot.levels,
                            regime or "–", confidence,
                        )
                        if prediction_flipped and not bot.with_position:
                            bot.emergency_sell(current_price, "Vorhersage DOWN")
                        bot.setup_grid(current_price, lower=lower, upper=upper)
                    # Dashboard immer aktualisieren wenn Prediction geprüft wurde
                    _log_bot_state(bot, current_price)

            # Zusammenfassung + Dashboard-State
            logger.info("─"*55)
            for bot in bots:
                price = prices.get(bot.symbol, 0)
                logger.info("%-12s Preis: %-10.4f %s", bot.symbol, price, bot.status())
                try:
                    from dashboard.db import update_grid_state
                    update_grid_state(
                        bot.symbol, price, bot.orders,
                        bot.range_pct, bot.investment,
                        bot.total_profit, bot.trade_count,
                        getattr(bot, "_last_prediction", ""),
                        predicted_low=bot._last_pred_low,
                        predicted_high=bot._last_pred_high,
                        confidence=bot._last_confidence,
                        regime=bot._last_regime,
                    )
                except Exception:
                    pass
            total_profit = sum(b.total_profit for b in bots)
            total_capital = sum(b.investment for b in bots)
            logger.info("GESAMT    Trades: %-5d Profit: %+.4f USDT | Kapital: %.2f USDT",
                        sum(b.trade_count for b in bots),
                        total_profit, total_capital)
            logger.info("─"*55)
            try:
                from dashboard.db import log_equity, update_capital
                log_equity(total_capital)
                update_capital(total_capital)
            except Exception:
                pass

        except Exception as e:
            logger.error("Fehler: %s", e)

        for _ in range(CHECK_INTERVAL):
            if not _running:
                break
            time.sleep(1)

    total_profit = sum(b.total_profit for b in bots)
    logger.info("Multi-Grid Bot gestoppt | Gesamt-Profit: %.4f USDT", total_profit)


if __name__ == "__main__":
    run()
