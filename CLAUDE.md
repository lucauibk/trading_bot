# Grid Trading Bot – Price Predictor

## Ziel
Implementiere ein Kursvorhersage-Modul für einen Grid Trading Bot.

## Projektstruktur
price_predictor/
├── data_fetcher.py       # OHLCV via ccxt
├── indicators.py         # ATR, Bollinger, RSI, VWAP, ADX
├── predictor.py          # Hauptklasse PricePredictor
├── grid_suggester.py     # Grid-Level aus Range berechnen
└── tests/
    ├── test_predictor.py
    └── test_indicators.py

## Technischer Stack
- Python 3.11+
- ccxt (Exchange-Daten)
- pandas_ta (Indikatoren)
- pytest (Tests)

## Gewünschtes Output-Interface
PricePredictor.predict() soll zurückgeben:
- predicted_low: float
- predicted_high: float  
- confidence: float (0.0–1.0)
- grid_levels: List[float] (10 Levels)
- regime: str ("ranging" | "trending" | "volatile")

## Regime-Logik
- ADX > 25 → trending → ATR-basierte Range (Faktor 2.0)
- ATR% > 3% → volatile → engeres Grid (Faktor 1.5)
- sonst → ranging → Bollinger Bands

## Regeln
- Immer TDD: erst Tests, dann Implementierung
- ATR immer als Fallback
- Kein Hardcoding von Exchange oder Symbol