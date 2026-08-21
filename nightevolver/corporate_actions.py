"""
NSE corporate actions, and the price adjustment they are needed for.

THIS MODULE EXISTS BECAUSE OF A MEASURED BUG
--------------------------------------------
`nse_prices.py` originally assumed the bhavcopy's `PrvsClsgPric` was the
exchange's corporate-action-ADJUSTED previous close, and said so
confidently in its docstring. That assumption is false, and the check
that caught it is worth recording:

    RELIANCE, 1:1 bonus, ex-date 2024-10-28
        PrvsClsgPric = 2655.70    ClsPric = 1334.35   ->  -49.76%

The field carries the raw prior close. Compounding that produced a
RELIANCE series with 49.9% annualised volatility (peers: ~20%) and a
-53.8% total return over a period the stock did not remotely have. A
single unadjusted bonus was most of both numbers.

It also slipped under a +-50% "data error" clamp, because a 1:1 bonus is
a -49.76% move - just inside the guard. A sanity filter tuned to a round
number will miss the most common corporate action there is.

WHAT ADJUSTMENT ACTUALLY REQUIRES
---------------------------------
Returns are corrected on the ex-date rather than back-adjusting a price
series, because it composes cleanly and keeps every other bar untouched:

    bonus a:b   holder ends with (a+b)/b shares
                adjusted_prev = prev * b/(a+b)
    split       face value X -> Y
                adjusted_prev = prev * Y/X
    dividend D  adjusted_close = close + D      (total-return convention)
    demerger    NOT COMPUTABLE from the announcement text

DEMERGERS ARE MASKED, NOT GUESSED. The value transferred to the
resulting entity is not in the corporate-actions feed, so the correct
factor is unknown. ITC's demerger (ex 2025-01-06) prints -8.09%, which
looks like an ordinary bad day and is not one - a holder lost nothing.
Injecting that as a real return would teach a model to fear a
non-event; inventing a factor would be worse. The ex-date return is
therefore dropped and counted, and the count is reported.

Verified against the live API: RELIANCE returns `Bonus 1:1` at
`exDate 28-Oct-2024`, alongside dividends and a 2023 demerger.
"""

from __future__ import annotations

import http.cookiejar
import json
import logging
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger("nightevolver.corpactions")

CA_URL = ("https://www.nseindia.com/api/corporates-corporateActions"
          "?index=equities&symbol={symbol}")

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache" / "nse_corpactions"

_UA = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-actions",
}

# "Bonus 1:1", "Bonus issue 2:1"
_BONUS_RE = re.compile(r"bonus\D*(\d+)\s*:\s*(\d+)", re.I)
# "Face Value Split From Rs 10 To Rs 2", and the real NSE wording
# "Face Value Split (Sub-Division) - From Rs 5/- Per Share To Re 1/- Per
# Share". The filler between the number and "To" is why this uses \D*
# rather than \s*: an earlier \s*to\s* version silently failed on the
# KOTAKBANK 2026-01-14 split, which then showed up as an unexplained
# -80.3% move. It was masked rather than trusted, which is the guard
# working - but a mask loses a real bar, so parse it properly.
_SPLIT_RE = re.compile(r"split.*?\bfrom\b\D*?([\d.]+)\D*?\bto\b\D*?([\d.]+)",
                       re.I | re.S)
# "Dividend - Rs 6 Per Share", "Interim Dividend Rs 5.50 Per Share"
_DIV_RE = re.compile(r"dividend\D*(?:rs\.?|re\.?)\s*([\d.]+)", re.I)
_DEMERGER_RE = re.compile(r"demerger|de-merger|arrangement|spin\s*-?\s*off", re.I)


@dataclass(frozen=True)
class CorporateAction:
    symbol: str
    ex_date: pd.Timestamp
    subject: str
    kind: str                 # bonus | split | dividend | demerger | other
    price_ratio: float = 1.0  # multiply the PREVIOUS close by this
    dividend: float = 0.0     # rupees per share, added back to close

    def __str__(self) -> str:
        return (f"{self.symbol} {self.ex_date.date()} {self.kind}: "
                f"{self.subject!r} ratio={self.price_ratio:.4f} div={self.dividend}")


def classify_action(symbol: str, subject: str,
                    ex_date: pd.Timestamp) -> CorporateAction:
    """Turn an announcement string into an adjustment.

    Order matters: a demerger that also mentions a ratio must classify as
    a demerger, because its ratio is not a price ratio.
    """
    s = (subject or "").strip()

    if _DEMERGER_RE.search(s):
        return CorporateAction(symbol, ex_date, s, "demerger")

    m = _BONUS_RE.search(s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        if a > 0 and b > 0:
            # a new shares for every b held -> (a+b)/b shares after
            return CorporateAction(symbol, ex_date, s, "bonus",
                                   price_ratio=b / (a + b))

    m = _SPLIT_RE.search(s)
    if m:
        old_fv, new_fv = float(m.group(1)), float(m.group(2))
        if old_fv > 0 and new_fv > 0:
            return CorporateAction(symbol, ex_date, s, "split",
                                   price_ratio=new_fv / old_fv)

    m = _DIV_RE.search(s)
    if m:
        return CorporateAction(symbol, ex_date, s, "dividend",
                               dividend=float(m.group(1)))

    return CorporateAction(symbol, ex_date, s, "other")


def _opener():
    cj = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    try:
        op.open(urllib.request.Request("https://www.nseindia.com/", headers=_UA),
                timeout=20).read()
    except Exception:
        pass          # cookie priming is best-effort; the API often works without
    return op


def fetch_corporate_actions(symbol: str, use_cache: bool = True,
                            max_attempts: int = 6,
                            opener=None) -> Optional[List[CorporateAction]]:
    """All announced corporate actions for one NSE symbol.

    RETURN VALUE CARRIES THREE STATES, and the difference is the whole
    point:

        [a, b, ...]  the endpoint answered and listed these actions
        []           the endpoint answered and has none for this symbol
        None         we could not get an answer

    Collapsing the last two into [] is what let an unadjusted bonus reach
    a price panel. adjust_returns refuses to proceed on None and accepts
    [], so a genuinely actionless symbol does not block a run while a
    failed fetch still does.
    """
    cp = CACHE_DIR / f"{symbol}.json"
    raw: Optional[bytes] = None
    if use_cache and cp.exists():
        try:
            raw = cp.read_bytes()
        except OSError:
            raw = None

    if raw is None:
        op = opener or _opener()
        url = CA_URL.format(symbol=urllib.request.quote(symbol))
        for attempt in range(max_attempts):
            try:
                with op.open(urllib.request.Request(url, headers=_UA),
                             timeout=25) as f:
                    raw = f.read()
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return []          # endpoint answered: no such symbol
            except Exception:
                pass
            if attempt < max_attempts - 1:
                time.sleep(min(5.0, 0.4 * (2 ** attempt)) * (0.5 + random.random()))
        if raw is None:
            # NOT []. An empty list means "the endpoint has no actions for
            # this symbol"; a failed fetch means "we do not know". Merging
            # them is what let a silently unadjusted -80% split bar reach
            # a price panel, which is the whole reason require_actions
            # exists. None is the honest answer and the guard reads it.
            logger.warning("[corpactions] could not fetch %s", symbol)
            return None
        if use_cache:
            try:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cp.write_bytes(raw)
            except OSError:
                pass

    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None                    # unreadable body is a failure, not a fact
    rows = payload if isinstance(payload, list) else payload.get("data", [])

    out: List[CorporateAction] = []
    for r in rows:
        ex = r.get("exDate")
        if not ex:
            continue
        try:
            ex_date = pd.Timestamp(pd.to_datetime(ex, format="%d-%b-%Y"))
        except (ValueError, TypeError):
            continue
        out.append(classify_action(r.get("symbol", symbol),
                                   r.get("subject", ""), ex_date))
    return sorted(out, key=lambda a: a.ex_date)


def fetch_all_corporate_actions(symbols: Sequence[str], use_cache: bool = True
                                ) -> Dict[str, Optional[List[CorporateAction]]]:
    op = _opener()
    out: Dict[str, Optional[List[CorporateAction]]] = {}
    for s in symbols:
        got = fetch_corporate_actions(s, use_cache=use_cache, opener=op)
        out[s] = got
        # None and [] read differently on purpose - see adjust_returns.
        logger.info("[corpactions] %-12s %s", s,
                    "FETCH FAILED" if got is None else f"{len(got)} actions")
    return out


def adjust_returns(close: pd.DataFrame, prev_close: pd.DataFrame,
                   actions: Dict[str, List[CorporateAction]],
                   require_actions: bool = True,
                   ) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Corporate-action-corrected daily returns.

    Returns (returns, masked) where `masked` is a boolean frame marking
    bars dropped because the correct adjustment is unknown (demergers,
    and unexplained extreme moves).

    Raises when `require_actions` and a symbol's fetch FAILED (None),
    because "we could not fetch it" and "there were none" must not be
    treated alike - silently reading the first as the second is how the
    original bug got in.

    An empty LIST is a different thing and is allowed through. NSE's
    corporate-actions endpoint only serves currently-listed symbols under
    their current name, so every point-in-time universe contains names it
    answers emptily for: a 2019 top-100 includes NIITTECH (now COFORGE),
    MCDOWELL-N (UNITDSPR), SRTRANSFIN (SHRIRAMFIN), L&TFH (LTF). Refusing
    those would mean dropping exactly the names that were later renamed,
    delisted or merged - reintroducing the survivorship bias that
    selecting the universe as-of a past date exists to remove.

    What protects those names is the residual mask below: any move still
    beyond 25% after adjustment is dropped as an action we do not know
    about. Measured over 2019-2026 on the six such names in a 2019
    top-100, that is 4 bars out of ~7,000 - and three of the four are
    real crashes (IndiaBulls 2019, COVID 2020) rather than actions, so
    the mask costs slightly more than it saves and both are negligible.
    """
    if require_actions:
        failed = [s for s in close.columns if actions.get(s, None) is None]
        if failed:
            raise RuntimeError(
                f"corporate-action fetch FAILED for {failed} (returned None, "
                f"not an empty list). An unadjusted bonus injects a ~-50% "
                f"fake return (see module docstring). Re-run to retry, or "
                f"pass require_actions=False if you accept that any "
                f"unexplained move beyond 25% will be masked rather than "
                f"corrected."
            )

    rets = close / prev_close.replace(0.0, np.nan) - 1.0
    masked = pd.DataFrame(False, index=close.index, columns=close.columns)

    applied = {"bonus": 0, "split": 0, "dividend": 0, "demerger": 0}
    for sym in close.columns:
        for act in (actions.get(sym) or []):
            if act.ex_date not in close.index:
                continue
            if act.kind == "demerger":
                masked.at[act.ex_date, sym] = True
                applied["demerger"] += 1
                continue
            prev = prev_close.at[act.ex_date, sym]
            cls = close.at[act.ex_date, sym]
            if not (np.isfinite(prev) and np.isfinite(cls)) or prev <= 0:
                continue
            adj_prev = prev * act.price_ratio
            adj_cls = cls + act.dividend
            if adj_prev > 0:
                rets.at[act.ex_date, sym] = adj_cls / adj_prev - 1.0
                if act.kind in applied:
                    applied[act.kind] += 1

    # Anything still extreme after adjustment is a corporate action we do
    # not know about or a bad row. Mask and COUNT it - do not silently
    # keep it, and do not silently drop it either.
    resid = rets.abs() > 0.25
    resid = resid & ~masked
    n_resid = int(resid.to_numpy().sum())
    if n_resid:
        for sym in close.columns:
            for d in close.index[resid[sym].to_numpy()]:
                logger.warning("[corpactions] unexplained %+.1f%% move %s %s "
                               "- masking (possible unlisted corporate action)",
                               rets.at[d, sym] * 100, sym, d.date())
        masked = masked | resid

    logger.info("[corpactions] applied bonus=%d split=%d dividend=%d | "
                "masked demerger=%d unexplained=%d",
                applied["bonus"], applied["split"], applied["dividend"],
                applied["demerger"], n_resid)

    rets = rets.mask(masked)
    return rets, masked
