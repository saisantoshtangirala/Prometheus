"""
News attention and tone, from GDELT (historical) and RSS (forward).

WHAT THIS ADDS. The system currently has ZERO text input. Every channel
is a number the exchange published. News is the mechanism by which most
large single-name moves actually happen, and an earnings surprise or a
regulatory action is visible in text hours before it is visible in a
daily bar - by which time the move has occurred and the "signal" is a
description of the past.

TWO SOURCES, FOR TWO DIFFERENT JOBS, and the distinction is not cosmetic:

  GDELT has HISTORY. It indexes global news back years and can be
  queried over a date range, so it can be backtested. This is the only
  free source here with that property, and it is therefore the only one
  that can be audited before being trusted.

  RSS has NO HISTORY. A feed returns its current window - typically
  15-35 items, hours to days old - and yesterday's items are simply
  gone. RSS can only be RECORDED FORWARD, exactly like the depth tape,
  and is worth nothing for backtesting until months have accumulated.

Conflating them would produce a feature that looks backtestable and is
not.

TWO MEASURED CAVEATS, both of which shape the code:

1. GDELT RATE-LIMITS HARD. Measured from this host: HTTP 429 on the
   first request, succeeding only after several seconds of backoff, and
   429 again immediately after. So every response is cached to disk and
   the retry schedule is deliberately slow. A tight loop here gets
   nothing and looks like an outage.

2. A STALE FEED RETURNS 200. Measured: Moneycontrol's business.xml
   served 15 well-formed items dated April 2024 during an August 2026
   session, while Business Standard served current ones. A dead feed
   does not error - it returns valid XML full of old news, which would
   enter the dataset as "no news today, every day". `feed_age_days`
   exists so that is detectable rather than silent.

ARTICLE COUNTS, NOT SENTIMENT SCORES. What is extracted is attention -
how many outlets are writing about this name today, relative to its own
recent baseline - plus GDELT's own tone field where present. Running a
sentiment model over headlines was considered and rejected for now: the
model would be an unvalidated component inside an unvalidated feature,
and attention alone is measurable, cheap, and has a plausible mechanism.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger("nightevolver.news")

GDELT_DOC = "https://api.gdeltproject.org/api/v2/doc/doc"
CACHE_DIR = Path(__file__).parent.parent / "data" / "cache" / "news"

_UA = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}

# Indian financial feeds. Business Standard verified current; Moneycontrol
# verified STALE (April 2024 content in an August 2026 session) and kept
# only so the staleness check has a real subject.
RSS_FEEDS = {
    "business_standard_markets":
        "https://www.business-standard.com/rss/markets-106.rss",
    "moneycontrol_business":
        "https://www.moneycontrol.com/rss/business.xml",
}

FEATURE_NAMES = (
    "news_count",      # articles mentioning the name that day
    "news_count_z",    # vs its own trailing 60-day mean
    "news_tone",       # mean GDELT tone, negative = adverse coverage
)

ZSCORE_WINDOW = 60
_MIN_PERIODS = 20

# GDELT caps artlist at 250 records per query, so a long span must be
# chunked or it silently truncates - returning "no news" for the tail of
# any busy period, which is exactly backwards.
MAX_RECORDS = 250
CHUNK_DAYS = 7


def _cache_path(key: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
    return CACHE_DIR / f"{safe}.json"


def _gdelt_get(params: Dict[str, str], max_attempts: int = 6,
               use_cache: bool = True) -> Optional[Dict]:
    """One GDELT query, cached, with slow backoff.

    Backoff starts at 5s and grows: measured 429s on back-to-back
    requests from this host. Anything faster returns nothing and reads
    like the API is down.
    """
    key = urllib.parse.urlencode(sorted(params.items()))
    p = _cache_path(key)
    if use_cache and p.exists():
        try:
            return json.loads(p.read_text())
        except ValueError:
            pass

    url = f"{GDELT_DOC}?{urllib.parse.urlencode(params)}"
    for attempt in range(max_attempts):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=45) as f:
                payload = json.loads(f.read() or b"{}")
            if use_cache:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(payload))
            return payload
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503):
                logger.warning("[news] GDELT HTTP %s", e.code)
                return None
        except (urllib.error.URLError, OSError, ValueError, TimeoutError):
            pass
        if attempt < max_attempts - 1:
            time.sleep(5.0 * (attempt + 1))
    logger.warning("[news] GDELT gave up after %d attempts (rate limited)",
                   max_attempts)
    return None


def _parse_articles(payload: Optional[Dict]) -> pd.DataFrame:
    """GDELT artlist -> [date, tone] rows. Empty frame when absent.

    `seendate` is 'YYYYMMDDTHHMMSSZ'. Rows that fail to parse are
    dropped rather than defaulted to today, which would pile unrelated
    history onto the current bar.
    """
    arts = (payload or {}).get("articles") or []
    rows = []
    for a in arts:
        sd = str(a.get("seendate", ""))
        try:
            dt = datetime.strptime(sd[:15], "%Y%m%dT%H%M%S")
        except ValueError:
            continue
        tone = a.get("tone")
        try:
            tone = float(tone) if tone is not None else np.nan
        except (TypeError, ValueError):
            tone = np.nan
        rows.append((pd.Timestamp(dt.date()), tone))
    if not rows:
        return pd.DataFrame(columns=["date", "tone"])
    return pd.DataFrame(rows, columns=["date", "tone"])


def gdelt_daily_counts(query: str, start: str, end: Optional[str] = None,
                       use_cache: bool = True) -> pd.DataFrame:
    """Daily article count and mean tone for one query.

    Chunked into CHUNK_DAYS windows because artlist truncates at 250
    records. Without chunking a busy fortnight returns its first 250
    articles and reports zero for everything after - a silent,
    direction-dependent bias, since quiet names would look complete and
    busy ones truncated.
    """
    s = pd.Timestamp(start)
    e = pd.Timestamp(end or pd.Timestamp.today().normalize())
    frames = []
    cur = s
    while cur <= e:
        stop = min(cur + pd.Timedelta(days=CHUNK_DAYS), e)
        payload = _gdelt_get({
            "query": query, "mode": "artlist", "format": "json",
            "maxrecords": str(MAX_RECORDS),
            "startdatetime": f"{cur:%Y%m%d}000000",
            "enddatetime": f"{stop:%Y%m%d}235959",
        }, use_cache=use_cache)
        df = _parse_articles(payload)
        if not df.empty:
            frames.append(df)
        cur = stop + pd.Timedelta(days=1)

    if not frames:
        return pd.DataFrame(columns=["news_count", "news_tone"])
    allrows = pd.concat(frames, ignore_index=True)
    g = allrows.groupby("date")
    return pd.DataFrame({
        "news_count": g.size(),
        "news_tone": g["tone"].mean(),
    })


def _causal_z(frame: pd.DataFrame, window: int = ZSCORE_WINDOW) -> pd.DataFrame:
    """Trailing-only z-score. shift(1) before rolling - see delivery.py."""
    prior = frame.shift(1)
    mu = prior.rolling(window, min_periods=_MIN_PERIODS).mean()
    sd = prior.rolling(window, min_periods=_MIN_PERIODS).std(ddof=1)
    return (frame - mu) / sd.replace(0.0, np.nan)


def fetch_news_features(symbols: Sequence[str], start: str,
                        end: Optional[str] = None,
                        name_map: Optional[Dict[str, str]] = None,
                        use_cache: bool = True) -> Dict[str, pd.DataFrame]:
    """{feature_name: DataFrame[date x symbol]}.

    `name_map` supplies the search phrase per ticker, because a bare
    NSE symbol is a poor query: "INFY" is an abbreviation that appears
    rarely in prose, while "Infosys" is what articles actually say, and
    "ITC" collides with unrelated acronyms entirely. Without a mapping
    the ticker is used as-is and the counts will understate reality for
    some names and overstate it for others - a bias that varies BY NAME,
    which is worse than a uniform one because it survives normalisation.
    """
    name_map = name_map or {}
    syms = [str(s).upper().replace(".NS", "") for s in symbols]
    idx = pd.bdate_range(start, end or pd.Timestamp.today().normalize())

    counts = pd.DataFrame(index=idx, columns=syms, dtype=float)
    tones = pd.DataFrame(index=idx, columns=syms, dtype=float)

    for sym in syms:
        phrase = name_map.get(sym, sym)
        daily = gdelt_daily_counts(f'"{phrase}"', start, end, use_cache)
        if daily.empty:
            continue
        counts[sym] = daily["news_count"].reindex(idx)
        tones[sym] = daily["news_tone"].reindex(idx)

    # A day with no articles is genuinely zero attention, not missing -
    # unlike tone, which is undefined when nothing was written.
    counts = counts.fillna(0.0)
    return {
        "news_count": counts,
        "news_count_z": _causal_z(counts),
        "news_tone": tones,
    }


# ---------------------------------------------------------------------
# RSS: forward-recording only
# ---------------------------------------------------------------------

def parse_rss(raw: bytes) -> List[Dict]:
    """RSS bytes -> [{title, published}]. Empty list on unparseable XML."""
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return []
    out = []
    for it in root.findall(".//item"):
        title = (it.findtext("title") or "").strip()
        pub = (it.findtext("pubDate") or "").strip()
        ts = None
        for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z"):
            try:
                ts = datetime.strptime(pub, fmt)
                break
            except ValueError:
                continue
        out.append({"title": title, "published": ts})
    return out


def feed_age_days(items: Sequence[Dict],
                  now: Optional[datetime] = None) -> Optional[float]:
    """Age of the NEWEST item, in days. None when nothing is dated.

    THE STALENESS CHECK. Measured: Moneycontrol's business.xml served 15
    well-formed items dated April 2024 during an August 2026 session,
    with HTTP 200 and valid XML. A dead feed does not error - it returns
    old news, which enters the dataset as "nothing happened today", every
    day, forever. Callers must reject a feed older than a day or two
    rather than trusting a 200.
    """
    now = now or datetime.now(timezone.utc)
    dated = [i["published"] for i in items if i.get("published")]
    if not dated:
        return None
    newest = max(dated)
    if newest.tzinfo is None:
        newest = newest.replace(tzinfo=timezone.utc)
    return (now - newest).total_seconds() / 86400.0


def poll_feeds(feeds: Optional[Dict[str, str]] = None,
               max_age_days: float = 3.0,
               timeout: int = 25) -> Dict[str, List[Dict]]:
    """Fetch each feed, dropping any that is stale or unreadable.

    Returns only LIVE feeds. This is the forward-recording entry point -
    RSS carries no history, so its value accrues from being run daily
    from now on, exactly like the depth recorder.
    """
    feeds = feeds if feeds is not None else RSS_FEEDS
    out: Dict[str, List[Dict]] = {}
    for name, url in feeds.items():
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=timeout) as f:
                items = parse_rss(f.read())
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            logger.warning("[news] %s unreachable: %s", name, e)
            continue
        age = feed_age_days(items)
        if age is None:
            logger.warning("[news] %s has no parseable dates - skipping", name)
            continue
        if age > max_age_days:
            logger.warning("[news] %s is STALE: newest item %.1f days old "
                           "(HTTP 200 and valid XML, but dead)", name, age)
            continue
        logger.info("[news] %s: %d items, newest %.2f days old",
                    name, len(items), age)
        out[name] = items
    return out


def record_feeds(directory: Path, feeds: Optional[Dict[str, str]] = None,
                 now: Optional[datetime] = None) -> Path:
    """Append today's live headlines to a dated JSONL file.

    Append rather than overwrite: run twice in a day and the second run
    must not discard the first run's items, since a feed's window moves
    and the earlier items are already gone from the source.
    """
    now = now or datetime.now(timezone.utc)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"news_{now:%Y%m%d}.jsonl"
    live = poll_feeds(feeds)
    n = 0
    with open(path, "a", encoding="utf-8") as fh:
        for feed, items in live.items():
            for it in items:
                pub = it.get("published")
                fh.write(json.dumps({
                    "feed": feed,
                    "title": it.get("title", ""),
                    "published": pub.isoformat() if pub else None,
                    "recorded": now.isoformat(),
                }) + "\n")
                n += 1
    logger.info("[news] recorded %d headlines from %d live feeds -> %s",
                n, len(live), path)
    return path
