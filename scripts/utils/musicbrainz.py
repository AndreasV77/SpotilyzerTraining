"""
musicbrainz.py
===============
Gemeinsame MusicBrainz-Hilfsfunktionen: ISRC-Lookup via Artist+Title-Suche.

Rate-Limit: 1 req/s (MusicBrainz API-Policy) — mb_get() schläft daher 1.1s
pro Call. Wird von scout_kworb.py und enrich_isrc.py genutzt.
"""

import time
import logging
from typing import Optional

import requests

MB_API_BASE   = "https://musicbrainz.org/ws/2"
MB_USER_AGENT = "SpotilyzerTraining/1.0 (github.com/AndreasV77/SpotilyzerTraining)"

_fallback_logger = logging.getLogger(__name__)


def mb_get(
    endpoint: str,
    params: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[dict]:
    """MusicBrainz API GET. Rate-Limit: 1 req/s."""
    log = logger or _fallback_logger
    url = f"{MB_API_BASE}/{endpoint}"
    headers = {
        "User-Agent": MB_USER_AGENT,
        "Accept":     "application/json",
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        time.sleep(1.1)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 404:
            return None
        log.warning(f"MusicBrainz HTTP {resp.status_code}: {url}")
        if resp.status_code == 503:
            time.sleep(10)
        return None
    except requests.exceptions.RequestException as e:
        log.error(f"MusicBrainz Fehler: {e}")
        return None


def get_isrc_by_artist_title(
    artist: str,
    title: str,
    cache: dict,
    logger: Optional[logging.Logger] = None,
) -> Optional[str]:
    """
    ISRC-Lookup via MusicBrainz Artist+Title-Suche.
    Cache-Key: "{artist_lower}|||{title_lower}"

    Zwei API-Calls:
      1. /recording?query=recording:"{title}" AND artist:"{artist}" → MBID
      2. /recording/{mbid}?inc=isrcs → ISRC
    """
    cache_key = f"{artist.lower()}|||{title.lower()}"
    if cache_key in cache:
        return cache[cache_key]

    # Schritt 1: Artist+Title → MBID
    data = mb_get("recording", params={
        "query": f'recording:"{title}" AND artist:"{artist}"',
        "fmt":   "json",
        "limit": 3,
    }, logger=logger)

    mbid = None
    if data and data.get("recordings"):
        mbid = data["recordings"][0]["id"]

    if not mbid:
        cache[cache_key] = None
        return None

    # Schritt 2: MBID → ISRC
    data = mb_get(f"recording/{mbid}", params={"inc": "isrcs", "fmt": "json"}, logger=logger)

    isrc = None
    if data:
        isrcs = data.get("isrcs", [])
        if isrcs:
            isrc = isrcs[0]

    cache[cache_key] = isrc
    return isrc
