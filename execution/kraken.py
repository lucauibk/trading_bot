"""
KrakenBroker – live trading on Kraken via ccxt.

Key improvements over the old LiveGridBot:
- postOnly=True on all limit orders (guarantees maker fee 0.16%)
- client_order_id via uuid4 for idempotency
- Precision rounding via market.info (lot_size, price precision)
- Retry with exponential backoff on network errors
- Fill reconciliation via fetch_my_trades(since=ts) + fetch_open_orders
  (replaces fragile fetch_closed_orders(limit=20))
"""

import logging
import time
import uuid
from typing import Dict, List, Optional

import ccxt

from core.strategy import Fill, Order
from execution.broker import Broker, BrokerOrder

logger = logging.getLogger(__name__)

KRAKEN_FEE = 0.0016
_MAX_RETRIES = 3
_BACKOFF_BASE = 1.5  # seconds


def _with_retry(fn, *args, **kwargs):
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            last_exc = e
            sleep = _BACKOFF_BASE ** attempt
            logger.warning("Kraken network error (attempt %d): %s – retry in %.1fs", attempt + 1, e, sleep)
            time.sleep(sleep)
        except ccxt.ExchangeError as e:
            raise  # don't retry logic errors
    raise last_exc


class KrakenBroker(Broker):

    def __init__(self, api_key: str, api_secret: str):
        self._ex = ccxt.kraken({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "timeout": 15_000,
            "options": {"defaultType": "spot"},
        })
        self._markets: dict = {}
        self._client_to_exchange: Dict[str, str] = {}  # client_id → exchange order id

    def _load_markets(self):
        if not self._markets:
            self._markets = _with_retry(self._ex.load_markets)

    def _market(self, symbol: str) -> dict:
        self._load_markets()
        return self._markets.get(symbol, {})

    def round_qty(self, symbol: str, qty: float) -> float:
        m = self._market(symbol)
        precision = m.get("precision", {}).get("amount")
        if precision is not None:
            return float(self._ex.amount_to_precision(symbol, qty))
        return round(qty, 6)

    def round_price(self, symbol: str, price: float) -> float:
        m = self._market(symbol)
        precision = m.get("precision", {}).get("price")
        if precision is not None:
            return float(self._ex.price_to_precision(symbol, price))
        return round(price, 4)

    def place_limit(
        self,
        symbol: str,
        side: str,
        price: float,
        qty: float,
        post_only: bool = True,
        client_id: str = "",
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
    ) -> BrokerOrder:
        if not client_id:
            client_id = str(uuid.uuid4())
        price = self.round_price(symbol, price)
        qty = self.round_qty(symbol, qty)

        params = {"userref": int(client_id.replace("-", "")[:9], 16) % (2**31)}
        if post_only:
            params["oflags"] = "post"

        try:
            result = _with_retry(
                self._ex.create_limit_order, symbol, side, qty, price, params
            )
        except ccxt.InvalidOrder as e:
            logger.error("Invalid order %s %s qty=%.6f @ %.4f: %s", symbol, side, qty, price, e)
            raise

        exchange_id = result.get("id", "")
        self._client_to_exchange[client_id] = exchange_id
        logger.info("[LIVE] placed %s %s qty=%.6f @ %.4f id=%s", symbol, side, qty, price, exchange_id)

        return BrokerOrder(
            client_id=client_id,
            exchange_order_id=exchange_id,
            symbol=symbol,
            side=side,
            price=price,
            qty=qty,
            status="open",
            ts_placed=time.time(),
        )

    def cancel(self, client_id: str) -> bool:
        exchange_id = self._client_to_exchange.get(client_id)
        if not exchange_id:
            return False
        try:
            _with_retry(self._ex.cancel_order, exchange_id)
            logger.info("[LIVE] cancelled %s", exchange_id)
            return True
        except ccxt.OrderNotFound:
            return False
        except Exception as e:
            logger.warning("Cancel failed %s: %s", exchange_id, e)
            return False

    def cancel_all(self, symbol: str) -> int:
        open_orders = self.get_open_orders(symbol)
        count = 0
        for o in open_orders:
            try:
                _with_retry(self._ex.cancel_order, o.exchange_order_id)
                count += 1
            except Exception:
                pass
        return count

    def place_market(self, symbol: str, side: str, qty: float) -> Optional[BrokerOrder]:
        qty = self.round_qty(symbol, qty)
        try:
            result = _with_retry(self._ex.create_market_order, symbol, side, qty)
            exchange_id = result.get("id", "")
            client_id = str(uuid.uuid4())
            self._client_to_exchange[client_id] = exchange_id
            logger.info("[LIVE] market %s %s qty=%.6f id=%s", symbol, side, qty, exchange_id)
            return BrokerOrder(
                client_id=client_id, exchange_order_id=exchange_id,
                symbol=symbol, side=side, price=0.0, qty=qty,
                status="filled", ts_placed=time.time(),
            )
        except Exception as e:
            logger.error("place_market failed %s %s: %s", symbol, side, e)
            return None

    def reconcile_fills(self, since_ts: float) -> List[Fill]:
        """Fetch fills since timestamp, resolving exchange_order_id → client_id via tracked map."""
        fills = []
        since_ms = int(since_ts * 1000)
        # Build reverse map: exchange_id → client_id
        exchange_to_client = {v: k for k, v in self._client_to_exchange.items()}
        try:
            trades = _with_retry(self._ex.fetch_my_trades, None, since=since_ms, limit=200)
            for t in trades:
                exchange_order_id = t.get("order", "")
                client_id = exchange_to_client.get(exchange_order_id, "")
                fills.append(Fill(
                    client_id=client_id,
                    symbol=t["symbol"],
                    side=t["side"],
                    price=float(t["price"]),
                    qty=float(t["amount"]),
                    fee=float(t.get("fee", {}).get("cost", 0) or 0),
                    ts=float(t["timestamp"]) / 1000,
                    exchange_order_id=exchange_order_id,
                ))
        except Exception as e:
            logger.warning("reconcile_fills error: %s", e)
        return fills

    def get_open_orders(self, symbol: str) -> List[BrokerOrder]:
        try:
            orders = _with_retry(self._ex.fetch_open_orders, symbol)
            result = []
            for o in orders:
                result.append(BrokerOrder(
                    client_id="",
                    exchange_order_id=o.get("id", ""),
                    symbol=symbol,
                    side=o["side"],
                    price=float(o["price"] or 0),
                    qty=float(o["amount"]),
                    filled_qty=float(o.get("filled", 0) or 0),
                    status="open",
                    ts_placed=float(o.get("timestamp", 0) or 0) / 1000,
                ))
            return result
        except Exception as e:
            logger.warning("get_open_orders error for %s: %s", symbol, e)
            return []

    def get_balance(self, currency: str = "USD") -> float:
        try:
            balance = _with_retry(self._ex.fetch_balance)
            return float(balance["free"].get(currency, 0.0))
        except Exception as e:
            logger.error("get_balance error: %s", e)
            return 0.0
