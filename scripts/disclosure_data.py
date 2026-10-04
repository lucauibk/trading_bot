"""
research/10 — Loader für offengelegte Trades (Senate-PTRs + SEC Form 4) und Kurse.

Quellen (kostenlos, Cache unter data/research_cache/, gitignored):
- Senate: timothycarambat/senate-stock-watcher-data, data/transaction_report_for_*.json
  (Feld `date_recieved` = Meldedatum). Abdeckung 2012-07 → 2021-03.
- Insider: SEC „Insider Transactions Data Sets" (Form 345), quartalsweise ZIPs.
- Kurse: yfinance (Close bereinigt, Volume, Stock Splits), parquet-Cache.
- Faktoren: Fama-French 5 (2×3) + Momentum, daily.

Läuft im Research-venv (python3.11 -m venv .venv-research; pip install pandas numpy yfinance):
  .venv-research/bin/python scripts/disclosure_data.py --fetch   # Rohdaten laden (einmalig)
  .venv-research/bin/python scripts/disclosure_data.py           # Event-Zählung
"""
import argparse, io, json, logging, re, sys, time, urllib.request, zipfile
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("disclosure")

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "research_cache"
SEC_DIR = CACHE / "sec"
SENATE_DIR = CACHE / "congress" / "daily"
FF_DIR = CACHE / "ff"
PX_CACHE = CACHE / "prices.pkl"

EVENT_COLS = ["event_id", "person", "ticker", "tx_date", "filing_date",
              "value_usd", "tx_price"]

# research/10 §2 — fixierte Filter
SENATE_MIN_AMOUNT = 15_001
INSIDER_MIN_PRICE = 5.0
INSIDER_CLUSTER_DAYS = 30
INSIDER_CLUSTER_MIN_USD = 100_000
INSIDER_COOLDOWN_DAYS = 90
INSIDER_MAX_FILING_DELAY_DAYS = 30  # Nachtrag 2026-10-04, vor erster Rendite
TICKER_RE = re.compile(r"^[A-Z]{1,5}([.-][A-Z])?$")


# ---------------------------------------------------------------------------
# Senate (H10a)
# ---------------------------------------------------------------------------
def _clean_ticker(raw: str) -> str | None:
    if not isinstance(raw, str):
        return None
    t = re.sub(r"<[^>]+>", "", raw).strip().upper()
    if not t or t == "--":
        return None
    t = t.replace(".", "-")  # yfinance-Schreibweise (BRK.B → BRK-B)
    return t if TICKER_RE.match(t) else None


def _amount_low(s: str) -> float | None:
    m = re.match(r"\$([\d,]+)", (s or "").strip())
    return float(m.group(1).replace(",", "")) if m else None


def load_senate_purchases(directory: Path = SENATE_DIR) -> pd.DataFrame:
    """Ein Event je (Senator, Ticker, Meldung). Nur Aktienkäufe ≥ 15 001 $."""
    rows = []
    for f in sorted(directory.glob("transaction_report_for_*.json")):
        for rep in json.load(open(f)):
            person = f"{rep.get('first_name', '')} {rep.get('last_name', '')}".strip()
            filed = pd.to_datetime(rep.get("date_recieved"), format="%m/%d/%Y", errors="coerce")
            for tx in rep.get("transactions") or []:
                if tx.get("asset_type") != "Stock" or tx.get("type") != "Purchase":
                    continue
                tic = _clean_ticker(tx.get("ticker"))
                amt = _amount_low(tx.get("amount"))
                txd = pd.to_datetime(tx.get("transaction_date"), format="%m/%d/%Y", errors="coerce")
                if tic is None or amt is None or pd.isna(filed) or pd.isna(txd):
                    continue
                if amt < SENATE_MIN_AMOUNT:
                    continue
                rows.append({"person": person, "ticker": tic, "tx_date": txd,
                             "filing_date": filed, "value_usd": amt,
                             "ptr": rep.get("ptr_link", "")})
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=EVENT_COLS)
    df = df[df["filing_date"] >= df["tx_date"]]
    df = (df.groupby(["person", "ticker", "ptr"], as_index=False)
            .agg(tx_date=("tx_date", "min"), filing_date=("filing_date", "first"),
                 value_usd=("value_usd", "sum")))
    df["tx_price"] = np.nan
    df["event_id"] = (df["person"] + "|" + df["ticker"] + "|"
                      + df["tx_date"].dt.strftime("%Y-%m-%d") + "|BUY")
    df = df.drop_duplicates("event_id")
    return df[EVENT_COLS].sort_values("filing_date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Insider (H10b)
# ---------------------------------------------------------------------------
def _read_tsv(z: zipfile.ZipFile, name: str, cols: list[str]) -> pd.DataFrame:
    with z.open(name) as fh:
        return pd.read_csv(fh, sep="\t", usecols=cols, dtype=str,
                           quoting=3, on_bad_lines="skip")


def load_insider_filings(sec_dir: Path = SEC_DIR) -> pd.DataFrame:
    """Eine Zeile je Form-4-Accession mit Open-Market-Käufen von Directors/Officers."""
    parts = []
    for zpath in sorted(sec_dir.glob("*_form345.zip")):
        with zipfile.ZipFile(zpath) as z:
            sub = _read_tsv(z, "SUBMISSION.tsv", ["ACCESSION_NUMBER", "FILING_DATE",
                                                  "DOCUMENT_TYPE", "ISSUERCIK",
                                                  "ISSUERTRADINGSYMBOL"])
            tr = _read_tsv(z, "NONDERIV_TRANS.tsv", ["ACCESSION_NUMBER", "TRANS_DATE",
                                                     "TRANS_CODE", "TRANS_SHARES",
                                                     "TRANS_PRICEPERSHARE",
                                                     "TRANS_ACQUIRED_DISP_CD"])
            own = _read_tsv(z, "REPORTINGOWNER.tsv", ["ACCESSION_NUMBER", "RPTOWNERCIK",
                                                      "RPTOWNERNAME",
                                                      "RPTOWNER_RELATIONSHIP"])
        sub = sub[sub["DOCUMENT_TYPE"] == "4"]
        tr = tr[(tr["TRANS_CODE"] == "P") & (tr["TRANS_ACQUIRED_DISP_CD"] == "A")]
        tr = tr[tr["ACCESSION_NUMBER"].isin(sub["ACCESSION_NUMBER"])]
        if tr.empty:
            continue
        tr = tr.assign(shares=pd.to_numeric(tr["TRANS_SHARES"], errors="coerce"),
                       price=pd.to_numeric(tr["TRANS_PRICEPERSHARE"], errors="coerce"),
                       tx_date=pd.to_datetime(tr["TRANS_DATE"], format="%d-%b-%Y",
                                              errors="coerce"))
        tr = tr.dropna(subset=["shares", "price", "tx_date"])
        tr = tr[(tr["shares"] > 0) & (tr["price"] > 0)]
        tr["value"] = tr["shares"] * tr["price"]
        agg = tr.groupby("ACCESSION_NUMBER").agg(value_usd=("value", "sum"),
                                                 shares=("shares", "sum"),
                                                 tx_date=("tx_date", "min"))
        agg["tx_price"] = agg["value_usd"] / agg["shares"]

        rel = own["RPTOWNER_RELATIONSHIP"].fillna("")
        own = own[rel.str.contains("Director") | rel.str.contains("Officer")]
        owners = own.groupby("ACCESSION_NUMBER").agg(
            owner_ciks=("RPTOWNERCIK", lambda s: frozenset(s)),
            person=("RPTOWNERNAME", "first"))

        df = (agg.join(owners, how="inner")
                 .join(sub.set_index("ACCESSION_NUMBER"), how="inner"))
        parts.append(df.reset_index())
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True).drop_duplicates("ACCESSION_NUMBER")
    df["filing_date"] = pd.to_datetime(df["FILING_DATE"], format="%d-%b-%Y", errors="coerce")
    df["ticker"] = df["ISSUERTRADINGSYMBOL"].map(_clean_ticker)
    df = df.dropna(subset=["filing_date", "ticker"])
    delay = (df["filing_date"] - df["tx_date"]).dt.days
    df = df[(delay >= 0) & (delay <= INSIDER_MAX_FILING_DELAY_DAYS)]
    df = df[df["tx_price"] >= INSIDER_MIN_PRICE]
    return df.rename(columns={"ISSUERCIK": "issuer_cik"})[
        ["ACCESSION_NUMBER", "issuer_cik", "ticker", "person", "owner_ciks",
         "tx_date", "filing_date", "value_usd", "tx_price"]].reset_index(drop=True)


def build_insider_clusters(filings: pd.DataFrame) -> pd.DataFrame:
    """research/10 §2.2: ≥ 2 Meldungen mit disjunkten Owner-CIKs, Transaktionen
    innerhalb 30 Tagen, Summe ≥ 100k $. Event = Meldedatum der vollendenden Meldung.
    Danach 90 Tage Sperre je Issuer."""
    events = []
    win = pd.Timedelta(days=INSIDER_CLUSTER_DAYS)
    cooldown = pd.Timedelta(days=INSIDER_COOLDOWN_DAYS)
    for cik, g in filings.sort_values(["filing_date", "tx_date"]).groupby("issuer_cik"):
        blocked_until = pd.Timestamp.min
        seen = []  # Meldungen in Reihenfolge ihrer Sichtbarkeit (Meldedatum)
        for row in g.itertuples(index=False):
            seen.append(row)
            if row.filing_date <= blocked_until:
                continue
            recent = [r for r in seen if abs(r.tx_date - row.tx_date) <= win]
            owners: set = set()
            persons, value, n_units = [], 0.0, 0
            for r in recent:
                if owners & r.owner_ciks:
                    continue  # gleiche Person/gemeinsame Einreichung → keine neue Einheit
                owners |= r.owner_ciks
                persons.append(str(r.person))
                value += r.value_usd
                n_units += 1
            if n_units >= 2 and value >= INSIDER_CLUSTER_MIN_USD:
                # Kurs-Check nutzt die vollendende Meldung (Preis + Datum passen zusammen)
                events.append({"person": " + ".join(sorted(set(persons)))[:200],
                               "ticker": row.ticker, "tx_date": row.tx_date,
                               "filing_date": row.filing_date, "value_usd": value,
                               "tx_price": row.tx_price,
                               "event_id": f"{cik}|{row.ticker}|"
                                           f"{row.filing_date:%Y-%m-%d}|CLUSTER"})
                blocked_until = row.filing_date + cooldown
    df = pd.DataFrame(events)
    if df.empty:
        return pd.DataFrame(columns=EVENT_COLS)
    return df[EVENT_COLS].sort_values("filing_date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Kurse + Faktoren
# ---------------------------------------------------------------------------
def load_prices(tickers, start="2012-06-01", end=None, force=False) -> dict:
    """{'adj','close','volume','splits'}: DataFrames (Datum × Ticker).
    adj = split+dividenden-bereinigt (für Renditen); close/volume = nur split-bereinigt
    (für Dollar-Volumen und den Form-4-Preisabgleich). Inkrementeller Pickle-Cache."""
    import yfinance as yf
    fields = (("adj", "Adj Close"), ("close", "Close"), ("volume", "Volume"),
              ("splits", "Stock Splits"))
    cache = pd.read_pickle(PX_CACHE) if PX_CACHE.exists() and not force else None
    have = set(cache["adj"].columns) | cache.get("missing", set()) if cache else set()
    want = sorted(set(tickers) - have)
    if want:
        log.info("yfinance: %d neue Ticker", len(want))
        import time

        class _RateLimitSeen(logging.Handler):
            """yfinance meldet 429 nur im Log, nicht im Rückgabewert."""
            hit = False

            def emit(self, record):
                if "RateLimit" in record.getMessage() or "Too Many Requests" in record.getMessage():
                    _RateLimitSeen.hit = True

        logging.getLogger("yfinance").addHandler(_RateLimitSeen())
        frames, unresolved = [], set()
        for i in range(0, len(want), 100):
            chunk = want[i:i + 100]
            for attempt in range(7):
                _RateLimitSeen.hit = False
                d = yf.download(chunk, start=start, end=end, auto_adjust=False, actions=True,
                                progress=False, threads=4, group_by="column")
                if not _RateLimitSeen.hit:
                    break
                wait = 60 * 2 ** attempt
                log.warning("Yahoo Rate-Limit (Chunk %d), warte %ds", i // 100, wait)
                time.sleep(wait)
            else:
                unresolved |= set(chunk)  # nie als „missing" markieren → nächster Lauf
                continue
            if d.empty:
                continue
            if not isinstance(d.columns, pd.MultiIndex):  # Einzelticker
                d.columns = pd.MultiIndex.from_product([d.columns, chunk])
            frames.append(d)
            time.sleep(1)
        new = pd.concat(frames, axis=1) if frames else None
        parts = {}
        for key, col in fields:
            n = None
            if new is not None and col in new.columns.get_level_values(0):
                n = new[col].dropna(axis=1, how="all")
            if cache is not None and n is not None:
                parts[key] = cache[key].join(n, how="outer")
            elif n is not None:
                parts[key] = n
            else:
                parts[key] = cache[key] if cache is not None else pd.DataFrame()
        # Ticker ohne Daten (delistet/umbenannt) merken, nicht erneut anfragen
        parts["missing"] = (cache.get("missing", set()) if cache else set()) | (
            set(want) - set(parts["adj"].columns) - unresolved)
        if unresolved:
            log.warning("%d Ticker wegen Rate-Limit nicht geladen — Lauf wiederholen",
                        len(unresolved))
        cache = parts
        pd.to_pickle(cache, PX_CACHE)
    return cache


def load_ff_factors() -> pd.DataFrame:
    """FF5 (2×3) + UMD daily, in Dezimal (nicht Prozent)."""
    def _read(path):
        lines = open(path, encoding="latin-1").read().splitlines()
        start = next(i for i, l in enumerate(lines) if re.match(r"^\s*,", l))
        rows = []
        for l in lines[start + 1:]:
            p = [x.strip() for x in l.split(",")]
            if not re.match(r"^\d{8}$", p[0]):
                break
            rows.append(p)
        cols = [c.strip() for c in lines[start].split(",")][1:]
        df = pd.DataFrame([r[1:] for r in rows], columns=cols,
                          index=pd.to_datetime([r[0] for r in rows], format="%Y%m%d"))
        return df.astype(float) / 100.0
    ff5 = _read(next(FF_DIR.glob("F-F_Research_Data_5_Factors_2x3_daily*.csv")))
    mom = _read(next(FF_DIR.glob("F-F_Momentum_Factor_daily*.csv")))
    mom.columns = ["UMD"]
    return ff5.join(mom, how="inner")


# ---------------------------------------------------------------------------
# Rohdaten-Download (SEC, Senate, Fama-French) — Kurse lädt load_prices() selbst
# ---------------------------------------------------------------------------
SEC_URL = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{}"
SENATE_REPO = "timothycarambat/senate-stock-watcher-data"
FF_URLS = ["https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
           "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Momentum_Factor_daily_CSV.zip"]
# SEC verlangt einen User-Agent mit Kontakt (Platzhalter; ggf. eigene Adresse eintragen)
SEC_UA = "trading-bot-research contact@example.org"


def _get(url, ua="research"):
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def fetch_raw(first_year=2013, last_year=None):
    import urllib.error
    last_year = last_year or pd.Timestamp.today().year
    SEC_DIR.mkdir(parents=True, exist_ok=True)
    for y in range(first_year, last_year + 1):
        for q in range(1, 5):
            f = SEC_DIR / f"{y}q{q}_form345.zip"
            if f.exists() and f.stat().st_size > 0:
                continue
            try:
                f.write_bytes(_get(SEC_URL.format(f.name), ua=SEC_UA))
                log.info("SEC %s", f.name)
            except urllib.error.HTTPError as e:
                log.info("SEC %s nicht verfügbar (%s)", f.name, e.code)
            time.sleep(0.3)  # SEC-Fair-Access: < 10 req/s

    SENATE_DIR.mkdir(parents=True, exist_ok=True)
    tree = json.loads(_get(f"https://api.github.com/repos/{SENATE_REPO}/git/trees/master?recursive=1"))
    for x in tree["tree"]:
        p = x["path"]
        if p.startswith("data/") and p.endswith(".json"):
            out = SENATE_DIR / p[5:]
            if not out.exists():
                out.write_bytes(_get(f"https://raw.githubusercontent.com/{SENATE_REPO}/master/{p}"))
    log.info("Senate: %d Tagesdateien", len(list(SENATE_DIR.glob("*.json"))))

    FF_DIR.mkdir(parents=True, exist_ok=True)
    for url in FF_URLS:
        zipfile.ZipFile(io.BytesIO(_get(url))).extractall(FF_DIR)
    log.info("Fama-French: %s", sorted(x.name for x in FF_DIR.glob("*.csv")))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="Rohdaten herunterladen")
    if ap.parse_args().fetch:
        fetch_raw()
    sen = load_senate_purchases()
    print(f"Senate-Events: {len(sen)}  {sen['filing_date'].min()} → {sen['filing_date'].max()}")
    fil = load_insider_filings()
    print(f"Insider-Käufe (Form 4, Dir/Off, ≥5$): {len(fil)}")
    cl = build_insider_clusters(fil)
    print(f"Insider-Cluster-Events: {len(cl)}  {cl['filing_date'].min()} → {cl['filing_date'].max()}")
    ff = load_ff_factors()
    print(f"FF-Faktoren: {ff.index.min().date()} → {ff.index.max().date()}")


if __name__ == "__main__":
    sys.exit(main())
