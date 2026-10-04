"""Tests für research/10 (scripts/disclosure_data.py, scripts/disclosure_event_study.py).

Laufen im Research-venv:  .venv-research/bin/python -m pytest tests/test_disclosure_research.py
"""
import json

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from scripts.disclosure_data import (_clean_ticker, build_insider_clusters,
                                     load_senate_purchases)
from scripts.disclosure_event_study import (block_bootstrap_mean, calendar_portfolio,
                                            ff_alpha, resolve_positions)


# ---------------------------------------------------------------------------
# Senate-Parsing
# ---------------------------------------------------------------------------
def _tx(ticker="BA", asset="Stock", typ="Purchase", amount="$15,001 - $50,000",
        date="12/22/2020"):
    return {"transaction_date": date, "owner": "Spouse", "ticker": ticker,
            "asset_type": asset, "type": typ, "amount": amount}


def _write_report(tmp_path, transactions, received="12/31/2020"):
    rep = [{"first_name": "Pat", "last_name": "Roberts", "ptr_link": "x",
            "date_recieved": received, "transactions": transactions}]
    (tmp_path / "transaction_report_for_12_31_2020.json").write_text(json.dumps(rep))


def test_clean_ticker():
    assert _clean_ticker('<a href="https://finance.yahoo.com/q?s=BA">BA</a>') == "BA"
    assert _clean_ticker("BRK.B") == "BRK-B"
    assert _clean_ticker("--") is None
    assert _clean_ticker(float("nan")) is None
    assert _clean_ticker("NOT A TICKER") is None


def test_senate_filters_only_stock_purchases_above_min(tmp_path):
    _write_report(tmp_path, [
        _tx(),                                           # ok
        _tx(ticker="AAPL", asset="Stock Option"),        # Option → raus
        _tx(ticker="MSFT", typ="Sale (Full)"),           # Verkauf → raus
        _tx(ticker="PLUG", amount="$1,001 - $15,000"),   # zu klein → raus
        _tx(ticker="--", asset="Corporate Bond"),        # Anleihe → raus
    ])
    df = load_senate_purchases(tmp_path)
    assert list(df["ticker"]) == ["BA"]
    assert df["event_id"].iloc[0] == "Pat Roberts|BA|2020-12-22|BUY"


def test_senate_dedup_same_person_ticker_report(tmp_path):
    _write_report(tmp_path, [_tx(date="12/21/2020"), _tx(date="12/22/2020")])
    df = load_senate_purchases(tmp_path)
    assert len(df) == 1
    assert df["value_usd"].iloc[0] == 30_002
    assert df["tx_date"].iloc[0] == pd.Timestamp("2020-12-21")


def test_senate_drops_filing_before_transaction(tmp_path):
    _write_report(tmp_path, [_tx(date="01/05/2021")], received="12/31/2020")
    assert load_senate_purchases(tmp_path).empty


# ---------------------------------------------------------------------------
# Insider-Cluster
# ---------------------------------------------------------------------------
def _filing(acc, owners, tx, filed, value=60_000, cik="1", ticker="XYZ", price=10.0):
    return {"ACCESSION_NUMBER": acc, "issuer_cik": cik, "ticker": ticker,
            "person": "/".join(owners), "owner_ciks": frozenset(owners),
            "tx_date": pd.Timestamp(tx), "filing_date": pd.Timestamp(filed),
            "value_usd": value, "tx_price": price}


def test_cluster_needs_two_distinct_insiders():
    fil = pd.DataFrame([_filing("a1", ["A"], "2020-01-01", "2020-01-03"),
                        _filing("a2", ["A"], "2020-01-05", "2020-01-07")])
    assert build_insider_clusters(fil).empty


def test_joint_filing_counts_once():
    # Fonds + GP in einer Meldung, dann der GP erneut → keine zwei Einheiten
    fil = pd.DataFrame([_filing("a1", ["F", "GP"], "2020-01-01", "2020-01-03"),
                        _filing("a2", ["GP"], "2020-01-05", "2020-01-07")])
    assert build_insider_clusters(fil).empty


def test_cluster_event_on_completing_filing_date():
    fil = pd.DataFrame([_filing("a1", ["A"], "2020-01-01", "2020-01-03"),
                        _filing("a2", ["B"], "2020-01-10", "2020-01-14")])
    ev = build_insider_clusters(fil)
    assert len(ev) == 1
    assert ev["filing_date"].iloc[0] == pd.Timestamp("2020-01-14")
    assert ev["value_usd"].iloc[0] == 120_000


def test_cluster_value_threshold_and_window():
    small = pd.DataFrame([_filing("a1", ["A"], "2020-01-01", "2020-01-03", value=40_000),
                          _filing("a2", ["B"], "2020-01-10", "2020-01-14", value=40_000)])
    assert build_insider_clusters(small).empty
    far = pd.DataFrame([_filing("a1", ["A"], "2020-01-01", "2020-01-03"),
                        _filing("a2", ["B"], "2020-03-01", "2020-03-03")])
    assert build_insider_clusters(far).empty


def test_cluster_cooldown_blocks_repeat():
    fil = pd.DataFrame([_filing("a1", ["A"], "2020-01-01", "2020-01-03"),
                        _filing("a2", ["B"], "2020-01-10", "2020-01-14"),
                        _filing("a3", ["C"], "2020-01-12", "2020-01-16"),   # in Sperre
                        _filing("a4", ["D"], "2020-06-01", "2020-06-03"),
                        _filing("a5", ["E"], "2020-06-02", "2020-06-04")])  # nach Sperre
    ev = build_insider_clusters(fil)
    assert list(ev["filing_date"]) == [pd.Timestamp("2020-01-14"), pd.Timestamp("2020-06-04")]


# ---------------------------------------------------------------------------
# Positionen, Portfolio, Statistik
# ---------------------------------------------------------------------------
@pytest.fixture
def px_cal():
    cal = pd.bdate_range("2020-01-01", periods=60)
    adj = pd.DataFrame({"XYZ": np.linspace(10, 16, 60), "SPY": 100.0}, index=cal)
    vol = pd.DataFrame({"XYZ": 1e6, "SPY": 1e6}, index=cal)
    splits = pd.DataFrame(0.0, index=cal, columns=["XYZ", "SPY"])
    return {"adj": adj, "close": adj.copy(), "volume": vol, "splits": splits}, cal


def _events(filed, tx="2020-01-02", price=10.1):
    return pd.DataFrame([{"event_id": "e", "person": "p", "ticker": "XYZ",
                          "tx_date": pd.Timestamp(tx), "filing_date": pd.Timestamp(filed),
                          "value_usd": 1e5, "tx_price": price}])


def test_entry_strictly_after_filing(px_cal):
    px, cal = px_cal
    valid, _, _ = resolve_positions(_events("2020-01-15"), px, cal, 5, check_price=False)
    assert cal[valid["i0"].iloc[0]] == pd.Timestamp("2020-01-16")
    assert valid["i1"].iloc[0] - valid["i0"].iloc[0] == 5


def test_price_mismatch_drops_reused_ticker(px_cal):
    px, cal = px_cal
    ok, _, _ = resolve_positions(_events("2020-01-15", price=10.1), px, cal, 5, check_price=True)
    bad, _, r = resolve_positions(_events("2020-01-15", price=50.0), px, cal, 5, check_price=True)
    assert len(ok) == 1 and bad.empty and r["preis_mismatch"] == 1


def test_illiquid_dropped(px_cal):
    px, cal = px_cal
    px["volume"]["XYZ"] = 10.0  # ~100 $/Tag
    valid, n_cov, r = resolve_positions(_events("2020-02-14"), px, cal, 5, check_price=False)
    assert valid.empty and n_cov == 1 and r["illiquide"] == 1


def test_calendar_portfolio_charges_costs(px_cal):
    px, cal = px_cal
    px["adj"]["XYZ"] = 10.0  # flach → Rendite nur aus Kosten
    pos = pd.DataFrame([{"ticker": "XYZ", "i0": 10, "i1": 15}])
    port, cnt = calendar_portfolio(pos, px, cal, 0.0035)
    assert port.iloc[11] == pytest.approx(-0.0035)
    assert port.iloc[15] == pytest.approx(-0.0035)
    assert port.iloc[12:15].abs().max() == 0
    assert np.isnan(port.iloc[16]) and cnt.iloc[16] == 0


def test_bootstrap_noise_includes_zero():
    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(0, 0.01, 2000))
    ci = block_bootstrap_mean(x, n_iter=2000)
    assert ci["lo"] < 0 < ci["hi"] and not ci["excl0"]


def test_ff_alpha_recovers_known_alpha():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2015-01-01", periods=1500)
    ff = pd.DataFrame(rng.normal(0, 0.01, (1500, 6)), index=idx,
                      columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"])
    ff["RF"] = 0.0001
    port = ff["RF"] + 0.0004 + 1.2 * ff["Mkt-RF"] + 0.5 * ff["SMB"] + rng.normal(0, 0.002, 1500)
    a = ff_alpha(port, ff)
    assert a["alpha_d"] == pytest.approx(0.0004, abs=1.5e-4)
    assert a["beta_mkt"] == pytest.approx(1.2, abs=0.05)
    assert a["t"] > 2
