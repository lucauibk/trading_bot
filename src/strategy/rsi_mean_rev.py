"""
RSI Mean Reversion: kauft bei überverkauft + Aufwärtstrend.
"""

import pandas as pd
from src.strategy.base_strategy import BaseStrategy, Signal, TradeSignal
from src.data.processor import add_indicators


class RSIMeanRevStrategy(BaseStrategy):

    def name(self) -> str:
        return "RSI_MeanRev"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        df = add_indicators(df, self.params)
        df.dropna(inplace=True)
        if len(df) < 2:
            return TradeSignal(Signal.NONE, 0, 0, 0)

        curr = df.iloc[-1]
        prev = df.iloc[-2]
        price = curr["close"]
        atr   = curr["atr"]

        oversold  = self.params.get("rsi_oversold", 30)
        rsi_bounce = prev["rsi"] < oversold and curr["rsi"] >= oversold
        in_uptrend = price > curr["ema_50"]
        near_bb    = price <= curr["bb_lower"] * 1.02

        if rsi_bounce and in_uptrend and near_bb:
            sl = self.get_stop_loss(price, Signal.LONG, atr)
            tp = self.get_take_profit(price, sl)
            return TradeSignal(Signal.LONG, price, sl, tp,
                               f"RSI bounce {prev['rsi']:.1f}→{curr['rsi']:.1f}")

        return TradeSignal(Signal.NONE, 0, 0, 0)

    def should_exit(self, df: pd.DataFrame, direction: Signal) -> bool:
        if direction != Signal.LONG:
            return False
        df = add_indicators(df, self.params)
        df.dropna(inplace=True)
        if df.empty:
            return False
        return df.iloc[-1]["rsi"] >= self.params.get("rsi_exit", 65)
