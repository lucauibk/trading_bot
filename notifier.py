import logging
import threading
import requests

import config

logger = logging.getLogger(__name__)


def _send(text: str):
    """Sendet Telegram-Nachricht in Daemon-Thread – blockiert den Bot nicht."""
    if not config.TELEGRAM_TOKEN or not config.TELEGRAM_CHAT_ID:
        return

    def _do_send():
        try:
            url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
            requests.post(
                url,
                json={"chat_id": config.TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
                timeout=8,
            )
        except Exception as e:
            logger.warning("Telegram-Fehler: %s", e)

    t = threading.Thread(target=_do_send, daemon=True)
    t.start()


def notify_trade_open(symbol: str, direction: str, entry: float, stop: float, tp: float, qty: float):
    mode = "📄 PAPER" if config.PAPER_TRADING else "🔴 LIVE"
    emoji = "📈" if direction == "LONG" else "📉"
    text = (
        f"{mode} | {emoji} <b>Trade eröffnet</b>\n"
        f"Symbol:  <code>{symbol}</code>\n"
        f"Richtung: <b>{direction}</b>\n"
        f"Entry:   <code>{entry:.4f}</code>\n"
        f"Stop:    <code>{stop:.4f}</code>\n"
        f"Target:  <code>{tp:.4f}</code>\n"
        f"Menge:   <code>{qty:.6f}</code>"
    )
    _send(text)


def notify_trade_close(symbol: str, direction: str, entry: float, exit_price: float, pnl: float, reason: str):
    mode = "📄 PAPER" if config.PAPER_TRADING else "🔴 LIVE"
    emoji = "✅" if pnl > 0 else "❌"
    text = (
        f"{mode} | {emoji} <b>Trade geschlossen</b>\n"
        f"Symbol:  <code>{symbol}</code>\n"
        f"Richtung: <b>{direction}</b>\n"
        f"Entry:   <code>{entry:.4f}</code>\n"
        f"Exit:    <code>{exit_price:.4f}</code>\n"
        f"PnL:     <b>{pnl:+.2f} USDT</b>\n"
        f"Grund:   {reason}"
    )
    _send(text)


def notify_mtf_zone_reached(symbol: str, direction: str, zone_low: float, zone_high: float,
                            current_price: float, structure_level: float):
    arrow = "📈" if direction == "long" else "📉"
    _send(
        f"{arrow} <b>MTF Retest-Zone erreicht</b>\n"
        f"Symbol:    <code>{symbol}</code>\n"
        f"Richtung:  <b>{'LONG' if direction == 'long' else 'SHORT'}</b>\n"
        f"Zone:      <code>{zone_low:.4f} – {zone_high:.4f}</code>\n"
        f"Struktur:  <code>{structure_level:.4f}</code>\n"
        f"Preis:     <code>{current_price:.4f}</code>"
    )


def notify_mtf_target_reached(symbol: str, direction: str, entry: float, target: float,
                               current_price: float, auto_executed: bool):
    tag = "✅ (auto)" if auto_executed else "✅"
    _send(
        f"{tag} <b>MTF Ziel erreicht</b>\n"
        f"Symbol:  <code>{symbol}</code>\n"
        f"Entry:   <code>{entry:.4f}</code>\n"
        f"Target:  <code>{target:.4f}</code>\n"
        f"Preis:   <code>{current_price:.4f}</code>"
    )


def notify_mtf_invalidated(symbol: str, direction: str, entry: float, exit_price: float,
                            pnl: float, zone_low: float, zone_high: float):
    _send(
        f"❌ <b>MTF Setup invalidiert</b>\n"
        f"Symbol:  <code>{symbol}</code>\n"
        f"Entry:   <code>{entry:.4f}</code>\n"
        f"Exit:    <code>{exit_price:.4f}</code>\n"
        f"PnL:     <b>{pnl:+.2f} USDT</b>\n"
        f"Zone:    <code>{zone_low:.4f} – {zone_high:.4f}</code>"
    )


def notify_error(message: str):
    _send(f"⚠️ <b>Bot-Fehler</b>\n{message}")


def notify_startup(capital: float, symbols: list = None, strategy: str = "Grid + ML"):
    """Startup-Nachricht mit den echten Runtime-Symbolen (config.yaml), nicht
    dem stalen config.SYMBOLS, und der tatsächlichen Strategie (#53)."""
    mode = "Paper Trading" if config.PAPER_TRADING else "LIVE Trading"
    _send(
        f"🤖 <b>Trading Bot gestartet</b>\n"
        f"Modus:    {mode}\n"
        f"Kapital:  {capital:.2f} USDT\n"
        f"Symbole:  {', '.join(symbols) if symbols else '–'}\n"
        f"Strategie: {strategy}"
    )
