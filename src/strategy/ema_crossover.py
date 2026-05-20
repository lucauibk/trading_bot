"""
EMA 9/21 Crossover Strategie mit Trend- und Volume-Filter.
"""

import pandas as pd
from src.strategy.base_strategy import BaseStrategy, Signal, TradeSignal
from src.data.processor import add_indicators


class EMACrossoverStrategy(BaseStrategy):

    def name(self) -> str:
        return "EMA_Crossover"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        df = add_indicators(df, self.params)
        df.dropna(inplace=True)
        if len(df) < 2:
            return TradeSignal(Signal.NONE, 0, 0, 0)

        curr = df.iloc[-1]
        prev = df.iloc[-2]
        price = curr["close"]
        atr   = curr["atr"]

        # Crossover erkennen
        bull_cross = prev["ema_9"] <= prev["ema_21"] and curr["ema_9"] > curr["ema_21"]
        bear_cross = prev["ema_9"] >= prev["ema_21"] and curr["ema_9"] < curr["ema_21"]

        vol_ok = curr["volume"] > curr["volume_sma"] if self.params.get("volume_confirm") else True

        # LONG
        if bull_cross and price > curr["ema_50"] and vol_ok:
            sl = self.get_stop_loss(price, Signal.LONG, atr)
            tp = self.get_take_profit(price, sl)
            return TradeSignal(Signal.LONG, price, sl, tp, "EMA bull-cross + Trend + Vol")

        # SHORT (nur wenn nicht Paper-Only-Long)
        if bear_cross and price < curr["ema_50"] and vol_ok:
            sl = self.get_stop_loss(price, Signal.SHORT, atr)
            tp = self.get_take_profit(price, sl)
            return TradeSignal(Signal.SHORT, price, sl, tp, "EMA bear-cross + Trend + Vol")

        return TradeSignal(Signal.NONE, 0, 0, 0)

    def should_exit(self, df: pd.DataFrame, direction: Signal) -> bool:
        df = add_indicators(df, self.params)
        df.dropna(inplace=True)
        if len(df) < 2:
            return False
        curr, prev = df.iloc[-1], df.iloc[-2]
        if direction == Signal.LONG:
            return prev["ema_9"] >= prev["ema_21"] and curr["ema_9"] < curr["ema_21"]
        return prev["ema_9"] <= prev["ema_21"] and curr["ema_9"] > curr["ema_21"]
