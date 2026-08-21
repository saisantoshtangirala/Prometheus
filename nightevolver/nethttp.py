"""
The exception tuple every retry loop in this package must catch.

THE MEASURED FAILURE. A 2019-2023 equity backfill ran for roughly an
hour, cached 1,615 sessions, and then died outright:

    http.client.IncompleteRead: IncompleteRead(73728 bytes read,
                                               12597 more expected)

Every fetcher here wraps its urlopen in

    except (urllib.error.URLError, OSError, TimeoutError)

which looks exhaustive and is not. IncompleteRead's MRO is

    IncompleteRead -> HTTPException -> Exception

- it is NOT an OSError. So the single most likely transport failure when
pulling thousands of files from a throttling archive was the one class
the retry loop could not see, and it killed the whole run instead of
sleeping and trying again. The bug's shape is worth naming: the handler
was not missing, it was subtly incomplete, and it only showed up after
an hour of apparently healthy progress.

http.client.HTTPException also covers BadStatusLine, LineTooLong and
InvalidURL, all of which are transient against a loaded server and all
of which were equally uncaught.

WHAT IS DELIBERATELY NOT HERE. ssl.SSLError and socket.timeout are both
already OSError subclasses. urllib.error.HTTPError is a URLError
subclass but is handled separately at every call site, because the
status code decides whether to retry (403) or stop (404) - folding it in
here would collapse that distinction.
"""

from __future__ import annotations

import http.client
import urllib.error

#: Transient transport failures: sleep and retry, never abort the run.
TRANSIENT_NET_ERRORS = (
    urllib.error.URLError,
    http.client.HTTPException,
    OSError,
    TimeoutError,
)

__all__ = ["TRANSIENT_NET_ERRORS"]
