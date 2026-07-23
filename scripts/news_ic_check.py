"""
News-Sentiment Information-Coefficient Check (Stufe 1 des News/Sentiment-Plans,
siehe Memory project_profitability_ideas_backlog_2026-07-22.md, Punkt 4).

Billigster Discriminator, BEVOR irgendein LLM-Call oder Trading-Code angefasst wird
(analog zu scripts/funding_ic_check.py, das dasselbe für Funding-Rates gemacht hat):
testet ob News-Sentiment überhaupt Information über die Forward-Rendite (4h/24h) trägt.

Datenquelle: kostenlose RSS-Feeds (CoinDesk, Cointelegraph, Decrypt) statt CryptoPanic —
CryptoPanic verlangt inzwischen ein bezahltes Abo, das widerspricht dem "billigster Test
zuerst"-Prinzip dieses Plans. RSS hat kein Crowd-Vote, deshalb wird Sentiment über ein
kleines, transparentes Keyword-Lexikon geschätzt (positive minus negative Begriffe im
Titel+Summary) — bewusst kein LLM-Call in dieser Stufe, damit Stufe 1 wirklich kostenlos
bleibt. Stufe 2 (LLM-Sentiment über echte Headlines) bleibt der nächste Schritt, falls
dieses grobe Lexikon-Signal schon etwas zeigt.

Kill-Kriterium (vorab fixiert, siehe Memory):
- IC muss über BEIDE Horizonte (4h und 24h), gepoolt über alle 5 Symbole,
  p<0.05 UND vorzeichenstabil sein — sonst kein Signal (genau wie der Funding-Test
  starb: IC≈0.03-0.05, vorzeichen-instabil).

Kausalität strikt eingehalten:
- Sentiment-Feature zum Zeitpunkt t nutzt nur News mit published_at <= t
  (rollierende Summe der letzten NEWS_WINDOW_H Stunden, kein Blick in die Zukunft)

Bekannte Einschränkung: RSS-Feeds liefern nur die aktuell letzten ~30-50 Artikel pro
Quelle, keine Datumsbereichs-Abfrage — das Skript loggt ehrlich, wie weit die geladenen
Daten tatsächlich zurückreichen, statt das stillschweigend anzunehmen. Ein lokaler Cache
baut echte Tiefe über mehrere Läufe/Tage auf.

Nutzung:
  python3 scripts/news_ic_check.py --as-of 2026-07-22
  python3 scripts/news_ic_check.py --symbol SOL/USD --days 60 --as-of 2026-07-22
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("GRIDBOT_BACKTEST", "1")

import feedparser
import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr

from backtest.data import load_ohlcv

logger = logging.getLogger("news_ic")

SYMBOLS = ["SOL/USD", "ETH/USD", "AVAX/USD", "LINK/USD", "XRP/USD"]
FWD_HORIZONS_H = [4, 24]      # Forward-Rendite-Horizonte (Stunden)
NEWS_WINDOW_H = 24            # rollierendes Sentiment-Fenster (Stunden)

RSS_FEEDS = {
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cointelegraph": "https://cointelegraph.com/rss",
    "decrypt": "https://decrypt.co/feed",
}

# Bewusst konservativ: generische Wörter (bare "link"/"eth"/"sol") sind zu mehrdeutig
# ("click the link", "ether" in anderem Kontext) — vollständige Namen bevorzugt.
CURRENCY_KEYWORDS = {
    "SOL":  [r"\bsolana\b"],
    "ETH":  [r"\bethereum\b", r"\bether\b"],
    "AVAX": [r"\bavalanche\b", r"\bavax\b"],
    "LINK": [r"\bchainlink\b"],
    "XRP":  [r"\bxrp\b", r"\bripple\b"],
}

# Kleines, transparentes Krypto-Finanz-Lexikon — grob, aber nachvollziehbar (kein Black-Box-Modell).
POSITIVE_WORDS = [
    "surge", "surges", "rally", "rallies", "soar", "soars", "bullish", "adoption",
    "partnership", "upgrade", "breakout", "record high", "all-time high", "gain", "gains",
    "jump", "jumps", "approval", "approved", "integrat", "launch", "listing", "inflow",
    "outperform", "rebound", "recovery", "boost", "milestone",
]
NEGATIVE_WORDS = [
    "crash", "crashes", "plunge", "plunges", "dump", "dumps", "bearish", "hack", "hacked",
    "exploit", "lawsuit", "sell-off", "selloff", "ban", "banned", "fraud", "collapse",
    "liquidat", "outflow", "decline", "declines", "drop", "drops", "fear", "scam",
    "delist", "delisted", "breach", "hacker", "investigation", "sec sues",
]

_CACHE_DB = Path(__file__).resolve().parents[1] / "data" / "news_cache.db"


def _currency_code(symbol: str) -> str:
    return symbol.split("/")[0]


def _cache_conn() -> sqlite3.Connection:
    _CACHE_DB.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(_CACHE_DB)
    con.execute("""CREATE TABLE IF NOT EXISTS articles (
        id TEXT,
        currency TEXT,
        published_at TEXT,
        source TEXT,
        title TEXT,
        sentiment REAL,
        PRIMARY KEY (id, currency)
    )""")
    con.commit()
    return con


def _score_sentiment(text: str) -> float:
    """Positive minus negative Keyword-Treffer, case-insensitive Substring-Suche."""
    t = text.lower()
    pos = sum(t.count(w) for w in POSITIVE_WORDS)
    neg = sum(t.count(w) for w in NEGATIVE_WORDS)
    return float(pos - neg)


def _matches_currency(text: str, currency: str) -> bool:
    patterns = CURRENCY_KEYWORDS.get(currency, [])
    t = text.lower()
    return any(re.search(p, t) for p in patterns)


def _fetch_rss_articles() -> list:
    """Holt aktuelle Artikel aus allen RSS_FEEDS (keine Auth, keine Kosten)."""
    articles = []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; grid-bot-research/1.0)"}
    for source, url in RSS_FEEDS.items():
        try:
            # feedparser.parse(url) nutzt einen eigenen urllib-Fetch ohne Browser-UA —
            # manche Feeds (z.B. CoinDesk) liefern dann eine Fehlerseite statt XML.
            # Erst per requests mit UA holen, dann die Bytes parsen.
            resp = requests.get(url, timeout=15, headers=headers)
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
            if feed.bozo and not feed.entries:
                logger.warning("%s: RSS-Parse-Fehler (%s), keine Einträge", source, feed.bozo_exception)
                continue
        except Exception as e:
            logger.warning("%s: RSS-Fetch fehlgeschlagen: %s", source, e)
            continue

        for entry in feed.entries:
            title = entry.get("title", "")
            summary = entry.get("summary", "") or entry.get("description", "")
            published = entry.get("published_parsed") or entry.get("updated_parsed")
            if not published:
                continue
            import time as _time
            ts = pd.Timestamp(_time.strftime("%Y-%m-%dT%H:%M:%SZ", published), tz="UTC")
            articles.append({
                "id": entry.get("id") or entry.get("link") or title,
                "source": source,
                "published_at": ts,
                "title": title,
                "text": f"{title} {summary}",
            })
        logger.info("%s: %d Artikel geholt", source, len(feed.entries))
    return articles


def fetch_news(currency: str) -> pd.DataFrame:
    """Holt aktuelle Artikel aus allen RSS-Feeds, filtert auf `currency`, scored Sentiment
    per Keyword-Lexikon und akkumuliert im lokalen Cache.

    RSS kennt keine Datumsbereichs-Abfrage — jeder Lauf sieht nur die neuesten Artikel
    pro Quelle. Der Cache dient daher nicht dazu, Fetches zu sparen, sondern History
    über mehrere Läufe/Tage aufzubauen, statt sie zu verlieren.
    """
    con = _cache_conn()
    articles = _fetch_rss_articles()
    matched = [a for a in articles if _matches_currency(a["text"], currency)]
    if matched:
        rows = [
            (a["id"], currency, a["published_at"].isoformat(), a["source"],
             a["title"], _score_sentiment(a["text"]))
            for a in matched
        ]
        con.executemany(
            """INSERT OR REPLACE INTO articles
               (id, currency, published_at, source, title, sentiment)
               VALUES (?,?,?,?,?,?)""",
            rows,
        )
        con.commit()
    con.close()

    all_df = pd.read_sql(
        "SELECT * FROM articles WHERE currency = ?", _cache_conn(), params=(currency,)
    )
    if all_df.empty:
        return all_df
    all_df["published_at"] = pd.to_datetime(all_df["published_at"], utc=True)
    all_df = all_df.drop_duplicates(subset=["id"]).sort_values("published_at")
    return all_df


def build_causal_sentiment_feature(ohlcv: pd.DataFrame, news: pd.DataFrame) -> pd.Series:
    """Rollierende NEWS_WINDOW_H-Stunden-Summe des Sentiment-Scores je Artikel,
    kausal auf den OHLCV-Index gemappt (Kerze t sieht nur Artikel mit published_at <= t).
    """
    if news.empty:
        return pd.Series(0.0, index=ohlcv.index)

    news = news.set_index("published_at").sort_index()
    sent_series = news["sentiment"]
    idx_ts = sent_series.index

    sentiment_at_candle = []
    window = pd.Timedelta(hours=NEWS_WINDOW_H)
    for t in ohlcv.index:
        lo = t - window
        mask = (idx_ts > lo) & (idx_ts <= t)
        sentiment_at_candle.append(sent_series.loc[mask].sum())
    return pd.Series(sentiment_at_candle, index=ohlcv.index, dtype=float)


def compute_ic(symbol: str, days: int, as_of: pd.Timestamp | None) -> dict:
    logger.info("Lade OHLCV+News für %s (%d Tage)…", symbol, days)
    ohlcv = load_ohlcv(symbol, "1h", days)
    if as_of is not None:
        ohlcv = ohlcv[ohlcv.index <= as_of]

    currency = _currency_code(symbol)
    news = fetch_news(currency)
    if news.empty:
        logger.warning("%s: keine News-Daten (Keyword-Filter zu eng? RSS erreichbar?)", currency)
        return {}
    if as_of is not None:
        news = news[news["published_at"] <= as_of]
    if news.empty:
        logger.warning("%s: keine News-Daten vor --as-of", currency)
        return {}

    coverage_start = news["published_at"].min()
    coverage_days = (news["published_at"].max() - coverage_start).total_seconds() / 86400
    logger.info("%s: %d Artikel, Abdeckung %s -> %s (%.1f Tage) — angefragt: %d Tage",
                currency, len(news), coverage_start, news["published_at"].max(),
                coverage_days, days)
    if coverage_days < days * 0.5:
        logger.warning("%s: News-Abdeckung (%.1fd) deckt weniger als halb angefragtes "
                        "Fenster (%dd) ab — RSS-Tiefen-Limitierung, IC nur auf dem "
                        "abgedeckten Teil aussagekräftig. Cache wird über weitere Läufe "
                        "tiefer.", currency, coverage_days, days)

    sentiment = build_causal_sentiment_feature(ohlcv, news)

    result = {"symbol": symbol, "coverage_days": round(coverage_days, 1), "n_posts": len(news)}
    for h in FWD_HORIZONS_H:
        fwd_ret = ohlcv["close"].shift(-h) / ohlcv["close"] - 1.0
        df = pd.DataFrame({"sentiment": sentiment, "fwd_ret": fwd_ret}).dropna()
        # nur Zeilen behalten, in denen tatsächlich News-Abdeckung existierte
        df = df[df.index >= coverage_start]
        if len(df) < 50:
            result[f"n_{h}h"] = len(df)
            result[f"ic_{h}h"] = 0.0
            result[f"p_{h}h"] = 1.0
            continue

        # nicht-überlappende Stichprobe für ehrliche Signifikanz (wie funding_ic_check.py)
        indep = df.iloc[::h]
        # konstantes Sentiment (z.B. fast alles 0 bei dünner News-Abdeckung) macht
        # Spearman undefiniert (NaN) — das ist inhaltlich "kein Signal", kein Rechenfehler.
        if len(indep) >= 20 and indep["sentiment"].nunique() > 1:
            ic, p = spearmanr(indep["sentiment"], indep["fwd_ret"])
            if not np.isfinite(ic):
                ic, p = 0.0, 1.0
        else:
            ic, p = 0.0, 1.0
        result[f"n_{h}h"] = len(indep)
        result[f"ic_{h}h"] = round(float(ic), 4)
        result[f"p_{h}h"] = round(float(p), 4)

    return result


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--as-of", type=str, default=None,
                        help="ISO-Datum (YYYY-MM-DD): News+OHLCV danach verwerfen "
                             "(Dev/Vault-Schutz, siehe research/00-hypothesen.md). "
                             "Pflicht für Dev-Set-Läufe: --as-of 2026-07-22")
    args = parser.parse_args()

    as_of_ts = None
    if args.as_of:
        as_of_ts = pd.Timestamp(args.as_of, tz="UTC")
        logger.info("--as-of %s: Daten danach werden verworfen", args.as_of)

    symbols = [args.symbol] if args.symbol else SYMBOLS

    results = []
    for sym in symbols:
        try:
            r = compute_ic(sym, args.days, as_of_ts)
            if r:
                results.append(r)
        except Exception as e:
            logger.warning("%s fehlgeschlagen: %s", sym, e)

    if not results:
        print("\n❌ Keine Ergebnisse — News-Daten nicht verfügbar (RSS erreichbar? "
              "Keyword-Filter zu eng? Cache noch leer und aktuelle Artikel decken "
              "keine der 5 Coins ab?)")
        return

    print(f"\n{'═' * 90}")
    print(f"  NEWS-SENTIMENT INFORMATION COEFFICIENT (RSS + Keyword-Lexikon, rollierend {NEWS_WINDOW_H}h)")
    print(f"{'═' * 90}")
    header = f"  {'Symbol':<10} {'Artikel':>7} {'Cov(d)':>7}"
    for h in FWD_HORIZONS_H:
        header += f" {'IC_' + str(h) + 'h':>9} {'p_' + str(h) + 'h':>8}"
    print(header)
    print("  " + "-" * 86)

    both_signif_count = 0
    ic_by_horizon = {h: [] for h in FWD_HORIZONS_H}
    for r in results:
        row = f"  {r['symbol']:<10} {r['n_posts']:>7} {r['coverage_days']:>7}"
        signif_flags = []
        for h in FWD_HORIZONS_H:
            ic, p = r[f"ic_{h}h"], r[f"p_{h}h"]
            ic_by_horizon[h].append(ic)
            row += f" {ic:>9} {p:>8}"
            signif_flags.append(p < 0.05 and abs(ic) >= 0.03)
        print(row)
        if all(signif_flags) and len({np.sign(r[f"ic_{h}h"]) for h in FWD_HORIZONS_H}) == 1:
            both_signif_count += 1

    print()
    for h in FWD_HORIZONS_H:
        mean_abs = np.mean([abs(v) for v in ic_by_horizon[h]])
        print(f"  Ø |IC_{h}h| über alle Symbole: {mean_abs:.4f}")
    print(f"{'═' * 90}")

    print()
    if both_signif_count >= 3:
        print("  ✅ NEWS-SENTIMENT TRÄGT SIGNAL (beide Horizonte, vorzeichenstabil,")
        print("     ≥3 Symbole) — Stufe 2 (LLM-Sentiment) und Stufe 3 (Backtest) lohnen sich")
    elif both_signif_count >= 1:
        print("  ⚠️  SCHWACHES SIGNAL — grenzwertig, nur 1-2 Symbole zeigen stabiles IC")
        print("     → Stufe 2 nur mit gedämpfter Erwartung, Coverage-Limitierung prüfen")
    else:
        print("  ❌ KEIN SIGNAL — News-Sentiment (Keyword-Lexikon) trägt keine Information")
        print("     über Forward-Rendite in den getesteten Fenstern")
        print("     → Stufe 2 (LLM-Sentiment) nur bei Verdacht auf Lexikon-Grobheit testen,")
        print("     sonst Idee 4 als tot einstufen (wie Funding-IC).")
    print()


if __name__ == "__main__":
    main()
