"""
Static, hand-curated sector classification for the NSE names this
project's point-in-time universes tend to include.

WHY THIS IS NOT AN AUTHORITATIVE FEED, STATED PLAINLY. The bhavcopy
carries no sector field, and NSE's index-constituent pages (which would
be an authoritative, if coarse, proxy) are a live network dependency
this module deliberately avoids for something this exploratory. This
mapping was assembled by hand from each company's known primary line of
business as of ~2019-2024 and is a broad-brush GICS-style grouping, not
a licensed classification. It is fit for one purpose: grouping names for
a SECTOR-RELATIVE PAIRS test, where the cost of a wrong or coarse label
is a slightly noisier pairing, not a wrong trading decision - use it for
that, and rebuild it from NSE's own index constituents before trusting
it for anything with capital behind it.

Coverage is deliberately limited to names this project's 100-ticker
2019-as-of universe actually contains (see
scripts.run_evolved_walkforward.resolve_universe), not the whole
exchange. A name absent from SECTOR is unclassified, not "other" - the
caller must decide what that means rather than have this module guess.
"""

from __future__ import annotations

from typing import Dict, List, Optional

SECTOR: Dict[str, str] = {
    # Private banks
    "AXISBANK": "bank_private", "INDUSINDBK": "bank_private",
    "HDFCBANK": "bank_private", "ICICIBANK": "bank_private",
    "KOTAKBANK": "bank_private", "FEDERALBNK": "bank_private",
    "YESBANK": "bank_private",
    # PSU banks
    "SBIN": "bank_psu", "BANKBARODA": "bank_psu", "CANBK": "bank_psu",
    "PNB": "bank_psu", "BANKINDIA": "bank_psu", "UNIONBANK": "bank_psu",
    "INDIANB": "bank_psu", "ORIENTBANK": "bank_psu",
    # NBFC / housing finance / diversified financials
    "HDFC": "nbfc", "BAJFINANCE": "nbfc", "BAJAJFINSV": "nbfc",
    "IBULHSGFIN": "nbfc", "PFC": "nbfc", "RECLTD": "nbfc",
    "L&TFH": "nbfc", "SRTRANSFIN": "nbfc", "RELCAPITAL": "nbfc",
    "DHFL": "nbfc",
    # IT services
    "TCS": "it_services", "INFY": "it_services", "WIPRO": "it_services",
    "TECHM": "it_services", "TATAELXSI": "it_services",
    "NIITTECH": "it_services", "JUSTDIAL": "it_services",
    # Pharma
    "SUNPHARMA": "pharma", "AUROPHARMA": "pharma", "WOCKPHARMA": "pharma",
    "LUPIN": "pharma", "DRREDDY": "pharma", "CIPLA": "pharma",
    "BIOCON": "pharma",
    # Autos & auto ancillary
    "MARUTI": "auto", "M&M": "auto", "TATAMOTORS": "auto",
    "ASHOKLEY": "auto", "HEROMOTOCO": "auto", "EICHERMOT": "auto",
    "BAJAJ-AUTO": "auto", "ESCORTS": "auto", "BALKRISIND": "auto_ancillary",
    "CEATLTD": "auto_ancillary",
    # Metals & mining
    "TATASTEEL": "metals", "HINDALCO": "metals", "JSWSTEEL": "metals",
    "JINDALSTEL": "metals", "VEDL": "metals", "SAIL": "metals",
    "COALINDIA": "metals",
    # Oil, gas & energy
    "RELIANCE": "energy", "ONGC": "energy", "GAIL": "energy",
    "HINDPETRO": "energy", "BPCL": "energy", "IOC": "energy",
    "MGL": "energy", "IGL": "energy",
    # Power & infra utilities
    "NTPC": "power", "POWERGRID": "power", "BHEL": "capital_goods",
    # Cement & construction materials
    "ACC": "cement", "GRASIM": "cement",
    # Capital goods / engineering / construction
    "LT": "capital_goods", "SIEMENS": "capital_goods", "NCC": "construction",
    "BEML": "capital_goods",
    "DLF": "realty", "RELINFRA": "capital_goods",
    # FMCG / consumer staples
    "HINDUNILVR": "fmcg", "ITC": "fmcg", "DABUR": "fmcg",
    "COLPAL": "fmcg", "BRITANNIA": "fmcg", "UBL": "fmcg",
    "JUBLFOOD": "fmcg", "SUNTV": "media",
    # Consumer discretionary / retail
    "TITAN": "consumer_discretionary", "ASIANPAINT": "consumer_discretionary",
    "DMART": "retail", "PAGEIND": "retail", "PCJEWELLER": "retail",
    "ZEEL": "media",
    # Telecom
    "BHARTIARTL": "telecom",
    # Chemicals / diversified
    "SRF": "chemicals", "UPL": "chemicals", "PEL": "diversified",
    "JAICORPLTD": "diversified", "MCDOWELL-N": "consumer_discretionary",
    # Aviation / transport / ports
    "JETAIRWAYS": "aviation", "INDIGO": "aviation",
    "ADANIPORTS": "transport_infra",
    # Diversified conglomerates / large industrial groups
    "ADANIENT": "diversified", "DREDGECORP": "industrials",
}

# Names actually present in the 2019-as-of 100-name universe that this
# mapping does NOT classify - checked, not guessed. Kept explicit so a
# caller building the universe fresh gets a loud KeyError-shaped signal
# rather than a silently thinning sector group.
UNCLASSIFIED: List[str] = []


def sector_of(ticker: str) -> Optional[str]:
    """None (not 'other') for an unmapped ticker - see module docstring."""
    return SECTOR.get(ticker.upper())


def sector_groups(tickers: List[str], min_size: int = 3) -> Dict[str, List[str]]:
    """{sector: [tickers]} restricted to `tickers`, dropping any sector
    that ends up with fewer than `min_size` members - a pair or triple
    test needs at least a couple of same-sector names to pair within."""
    groups: Dict[str, List[str]] = {}
    for t in tickers:
        s = sector_of(t)
        if s is not None:
            groups.setdefault(s, []).append(t)
    return {s: ts for s, ts in groups.items() if len(ts) >= min_size}
