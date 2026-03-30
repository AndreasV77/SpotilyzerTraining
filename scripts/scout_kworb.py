"""
scout_kworb.py
==============
Scrapt Kworb.net _weekly_totals → ISRC via MusicBrainz → Deezer Track-ID.

Workflow:
  1. Kworb _weekly_totals für jeden Markt scrapen (pandas.read_html)
  2. Tracks filtern (Total >= min_streams ODER Pk <= 50 AND Wks >= 4)
  3. Tracks über alle Märkte aggregieren und deduplizieren
  4. ISRC via MusicBrainz (Artist+Title-Suche), mit lokalem Cache
  5. Deezer-Lookup via ISRC (/track/isrc:{isrc}), Fallback: Artist+Title-Suche
  6. ChartScore + Label berechnen
  7. Merge in datasets/kworb/tracks.jsonl

Output-Schema (pro Track):
  track_id       — Deezer Track-ID (Primärschlüssel)
  title, artist, album — Metadaten
  deezer_rank    — Deezer Popularity-Wert
  isrc           — ISRC (via MusicBrainz, oder null)
  chart_entries  — [{"country": "us", "peak_rank": 1, "weeks_on_chart": 50,
                     "t10_weeks": 20, "total_streams": 1000000000}, ...]
  chart_score    — max(MarketWeight * ChartScore) über alle Märkte
  label          — hit/mid
  robustness     — "validated"
  dataset        — "kworb"
  clusters       — ["charts_us", "charts_gb", ...]

Kworb-URL-Schema:
  https://kworb.net/spotify/country/{market}_weekly_totals.html
  Phase-1-Märkte: us, gb, de, jp, br, mx
  Phase-2-Märkte: fr, au, ca, it, se, nl

Filter-Kriterien (--min-streams, --quality-only):
  Default: Total >= 2_000_000 ODER (Pk <= 50 AND Wks >= 4)
  --quality-only: nur Pk <= 50 AND Wks >= 4 (ignoriert --min-streams)
"""

import sys
import time
import json
import math
import argparse
from pathlib import Path
from typing import Optional

import requests
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import setup_logging, load_paths_config, ensure_dir
from utils.metadata import merge_tracks, read_tracks

# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

KWORB_BASE      = "https://kworb.net/spotify/country"
DEEZER_API_BASE = "https://api.deezer.com"
MB_API_BASE     = "https://musicbrainz.org/ws/2"
MB_USER_AGENT   = "SpotilyzerTraining/1.0 (github.com/AndreasV77/SpotilyzerTraining)"

# Phase-1-Märkte (Session 5)
PHASE1_MARKETS = ["us", "gb", "de", "jp", "br", "mx"]
# Phase-2-Märkte (Session 6)
PHASE2_MARKETS = ["fr", "au", "ca", "it", "se", "nl"]

DEFAULT_MARKETS = PHASE1_MARKETS + PHASE2_MARKETS

# Markt-Gewichte
# Tier A (1.0):  us, gb — globale Referenzmärkte
# Tier B (0.85): de, jp, br, fr, au, ca — große Export-/Sprachmärkte
# Tier C (0.70): mx, it, se, nl — mittelgroße Märkte mit eigener Szene
# Tier D (0.50): alle anderen → Default via .get(..., 0.50)
MARKET_WEIGHTS = {
    "us": 1.0,
    "gb": 1.0,
    "de": 0.85,
    "jp": 0.85,
    "br": 0.85,
    "fr": 0.85,   # Phase 2
    "au": 0.85,   # Phase 2
    "ca": 0.85,   # Phase 2
    "mx": 0.70,
    "it": 0.70,   # Phase 2
    "se": 0.70,   # Phase 2
    "nl": 0.70,   # Phase 2
}

# Hit-Schwellenwerte je Tier (peak_rank <= threshold → Hit-Label)
# Tier D (0.50): top 10 in einem kleineren Markt → Hit (z.B. Norwegen, Polen)
HIT_THRESHOLDS = {
    1.0:  100,
    0.85:  50,
    0.70:  20,
    0.50:  10,
}

DEFAULT_MIN_STREAMS = 2_000_000

logger = None


# ══════════════════════════════════════════════════════════════════════════════
# KWORB SCRAPING
# ══════════════════════════════════════════════════════════════════════════════

def split_artist_title(combined: str) -> tuple[str, str]:
    """
    Spaltet "Artist - Title" in Artist und Titel.
    Nutzt erstes ' - ' (mit Leerzeichen) als Trennzeichen.
    """
    parts = combined.split(" - ", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return combined.strip(), ""


def clean_int(val) -> Optional[int]:
    """Bereinigt Integer-Wert aus pandas (kann NaN, float oder str sein)."""
    try:
        if pd.isna(val):
            return None
        return int(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def scrape_market(market: str, min_streams: int, quality_only: bool) -> list[dict]:
    """
    Scrapt _weekly_totals für einen Markt via pandas.read_html.

    Returns:
        Liste von Dicts mit artist, title, peak_rank, weeks_on_chart,
        t10_weeks, total_streams, pk_streams, market
    """
    url = f"{KWORB_BASE}/{market}_weekly_totals.html"
    logger.info(f"  Scraping {market}: {url}")

    try:
        resp = requests.get(url, headers={"User-Agent": MB_USER_AGENT}, timeout=30)
        resp.encoding = "utf-8"  # Kworb liefert UTF-8, requests erkennt es nicht immer
        time.sleep(0.5)  # Kworb höflich behandeln
        if resp.status_code != 200:
            logger.warning(f"  HTTP {resp.status_code} für {market}")
            return []

        from io import StringIO
        tables = pd.read_html(StringIO(resp.text), thousands=",")
        if not tables:
            logger.warning(f"  Keine Tabelle gefunden für {market}")
            return []

        df = tables[0]
        logger.info(f"  {market}: {len(df)} Zeilen, Spalten: {list(df.columns)}")

    except Exception as e:
        logger.error(f"  Scraping-Fehler für {market}: {e}")
        return []

    # Spaltennamen normalisieren — Kworb-Headers können leicht variieren
    col_map = {}
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if "artist" in col_lower and "title" in col_lower:
            col_map["artist_title"] = col
        elif col_lower == "wks":
            col_map["wks"] = col
        elif col_lower == "t10":
            col_map["t10"] = col
        elif col_lower == "pk":
            col_map["pk"] = col
        elif col_lower == "pkstreams":
            col_map["pkstreams"] = col
        elif col_lower == "total":
            col_map["total"] = col

    required = ["artist_title", "wks", "pk", "total"]
    missing = [k for k in required if k not in col_map]
    if missing:
        logger.warning(
            f"  Fehlende Spalten für {market}: {missing}. "
            f"Verfügbar: {list(df.columns)}"
        )
        return []

    tracks = []
    for _, row in df.iterrows():
        artist_title = str(row[col_map["artist_title"]])
        wks          = clean_int(row[col_map["wks"]])
        pk           = clean_int(row[col_map["pk"]])
        total        = clean_int(row[col_map["total"]])
        t10          = clean_int(row[col_map["t10"]]) if "t10" in col_map else None
        pk_streams   = clean_int(row[col_map["pkstreams"]]) if "pkstreams" in col_map else None

        if not wks or not pk or not total:
            continue

        # Filter
        passes_stream  = total >= min_streams
        passes_quality = pk <= 50 and wks >= 4

        if quality_only:
            if not passes_quality:
                continue
        else:
            if not passes_stream and not passes_quality:
                continue

        artist, title = split_artist_title(artist_title)
        if not title:
            logger.debug(f"  Konnte Artist/Titel nicht trennen: {artist_title!r}")
            continue

        tracks.append({
            "artist":         artist,
            "title":          title,
            "peak_rank":      pk,
            "weeks_on_chart": wks,
            "t10_weeks":      t10,
            "total_streams":  total,
            "pk_streams":     pk_streams,
            "market":         market,
        })

    logger.info(f"  {market}: {len(tracks)} Tracks nach Filter")
    return tracks


def aggregate_markets(
    all_market_tracks: dict[str, list[dict]],
    max_tracks: Optional[int] = None,
) -> dict[str, dict]:
    """
    Aggregiert Tracks über alle Märkte.
    Dedup-Key: (artist_lower, title_lower).

    Wenn max_tracks gesetzt: nach Dedup nach total_streams (max über alle Märkte)
    absteigend sortieren und auf max_tracks kappen.

    Returns:
        {key: {"artist": ..., "title": ..., "chart_entries": [...]}}
    """
    aggregated: dict[str, dict] = {}

    for market, tracks in all_market_tracks.items():
        for t in tracks:
            key = (t["artist"].lower(), t["title"].lower())

            if key not in aggregated:
                aggregated[key] = {
                    "artist":        t["artist"],
                    "title":         t["title"],
                    "chart_entries": [],
                    "_max_streams":  0,
                }

            aggregated[key]["chart_entries"].append({
                "country":        market,
                "peak_rank":      t["peak_rank"],
                "weeks_on_chart": t["weeks_on_chart"],
                "t10_weeks":      t["t10_weeks"],
                "total_streams":  t["total_streams"],
                "pk_streams":     t["pk_streams"],
            })
            # Höchsten Total-Streams-Wert über alle Märkte merken (für Sortierung)
            if t["total_streams"] and t["total_streams"] > aggregated[key]["_max_streams"]:
                aggregated[key]["_max_streams"] = t["total_streams"]

    if max_tracks and len(aggregated) > max_tracks:
        sorted_keys = sorted(
            aggregated.keys(),
            key=lambda k: aggregated[k]["_max_streams"],
            reverse=True,
        )
        aggregated = {k: aggregated[k] for k in sorted_keys[:max_tracks]}

    # Internes Sortierfeld entfernen
    for v in aggregated.values():
        v.pop("_max_streams", None)

    return aggregated


# ══════════════════════════════════════════════════════════════════════════════
# CHART SCORE & LABEL (identisch zu scout_spotify.py)
# ══════════════════════════════════════════════════════════════════════════════

def compute_chart_score_and_label(chart_entries: list[dict]) -> tuple[float, str]:
    """
    ChartScore = (1 / peak_rank) * log(weeks_on_chart + 1)
    HitScore   = max(MarketWeight * ChartScore)
    """
    best_score = 0.0
    is_hit = False

    for entry in chart_entries:
        weight = MARKET_WEIGHTS.get(entry["country"], 0.5)
        score  = weight * (1.0 / entry["peak_rank"]) * math.log(entry["weeks_on_chart"] + 1)
        best_score = max(best_score, score)

        threshold = HIT_THRESHOLDS.get(weight)
        if threshold and entry["peak_rank"] <= threshold:
            is_hit = True

    return round(best_score, 4), "hit" if is_hit else "mid"


# ══════════════════════════════════════════════════════════════════════════════
# MUSICBRAINZ
# ══════════════════════════════════════════════════════════════════════════════

def mb_get(endpoint: str, params: dict = None) -> Optional[dict]:
    """MusicBrainz API GET. Rate-Limit: 1 req/s."""
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
        logger.warning(f"MusicBrainz HTTP {resp.status_code}: {url}")
        if resp.status_code == 503:
            time.sleep(10)
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"MusicBrainz Fehler: {e}")
        return None


def get_isrc_by_artist_title(artist: str, title: str, cache: dict) -> Optional[str]:
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
    })

    mbid = None
    if data and data.get("recordings"):
        mbid = data["recordings"][0]["id"]

    if not mbid:
        cache[cache_key] = None
        return None

    # Schritt 2: MBID → ISRC
    data = mb_get(f"recording/{mbid}", params={"inc": "isrcs", "fmt": "json"})

    isrc = None
    if data:
        isrcs = data.get("isrcs", [])
        if isrcs:
            isrc = isrcs[0]

    cache[cache_key] = isrc
    return isrc


# ══════════════════════════════════════════════════════════════════════════════
# DEEZER (identisch zu scout_spotify.py)
# ══════════════════════════════════════════════════════════════════════════════

def deezer_get(endpoint: str, params: dict = None, delay: float = 0.3) -> Optional[dict]:
    """Deezer API GET."""
    url = f"{DEEZER_API_BASE}/{endpoint}"
    try:
        resp = requests.get(url, params=params, timeout=10)
        time.sleep(delay)
        if resp.status_code == 200:
            data = resp.json()
            if "error" not in data:
                return data
        elif resp.status_code == 429:
            logger.warning("Deezer Rate Limit, warte 5s...")
            time.sleep(5)
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Deezer Fehler: {e}")
        return None


def lookup_deezer_by_isrc(isrc: str) -> Optional[dict]:
    """Deezer-Track-Lookup via ISRC."""
    return deezer_get(f"track/isrc:{isrc}")


def lookup_deezer_by_search(artist: str, title: str) -> Optional[dict]:
    """Deezer-Track-Suche via Artist+Title (Fallback)."""
    first_artist = artist.split(",")[0].strip()

    # Strukturierte Suche
    data = deezer_get("search/track", params={
        "q":     f'artist:"{first_artist}" track:"{title}"',
        "limit": 5,
    })
    if data and data.get("data"):
        return data["data"][0]

    # Breitere Suche als Fallback
    data = deezer_get("search/track", params={
        "q":     f"{first_artist} {title}",
        "limit": 5,
    })
    if data and data.get("data"):
        return data["data"][0]

    return None


def build_track_dict(
    deezer_data: dict,
    isrc: Optional[str],
    chart_entries: list[dict],
    chart_score: float,
    label: str,
) -> dict:
    """Baut Track-Dict im SpotilyzerTraining-Format."""
    countries = sorted({e["country"] for e in chart_entries})
    return {
        "track_id":      deezer_data["id"],
        "title":         deezer_data.get("title", ""),
        "artist":        deezer_data.get("artist", {}).get("name", ""),
        "album":         deezer_data.get("album", {}).get("title", ""),
        "deezer_rank":   deezer_data.get("rank", 0),
        "isrc":          isrc,
        "chart_entries": chart_entries,
        "chart_score":   chart_score,
        "label":         label,
        "robustness":    "validated",
        "dataset":       "kworb",
        "clusters":      [f"charts_{c}" for c in countries],
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global logger

    parser = argparse.ArgumentParser(
        description="Kworb _weekly_totals → ISRC (MusicBrainz) → Deezer Track-IDs"
    )
    parser.add_argument(
        "--markets", nargs="+", default=DEFAULT_MARKETS,
        metavar="MARKET",
        help=f"Märkte (default: {' '.join(DEFAULT_MARKETS)})",
    )
    parser.add_argument(
        "--min-streams", type=int, default=DEFAULT_MIN_STREAMS,
        help=f"Minimale Gesamt-Streams (default: {DEFAULT_MIN_STREAMS:,})",
    )
    parser.add_argument(
        "--quality-only", action="store_true",
        help="Nur Pk <= 50 AND Wks >= 4 (ignoriert --min-streams)",
    )
    parser.add_argument(
        "--max-tracks", type=int, default=None,
        help="Nach Dedup auf Top N nach total_streams kappen (default: alle)",
    )
    parser.add_argument(
        "--skip-mb", action="store_true",
        help="MusicBrainz überspringen, direkt Deezer-Suche (schneller, kein ISRC)",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Output JSONL (default: datasets/kworb/tracks.jsonl)",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=100, metavar="N",
        help="Checkpoint alle N Tracks sichern (default: 100)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Kworb scrapen + Stats zeigen, keine Deezer/MusicBrainz-Calls",
    )
    args = parser.parse_args()

    logger = setup_logging("scout_kworb")
    paths  = load_paths_config()

    # ── Output-Pfad ──────────────────────────────────────────────────────────
    if args.output:
        output_path = args.output
    else:
        datasets_dir = paths.get("datasets", paths["data_root"] / "datasets")
        output_path  = datasets_dir / "kworb" / "tracks.jsonl"

    ensure_dir(output_path.parent)
    logger.info(f"Output: {output_path}")

    # ── ISRC-Cache ───────────────────────────────────────────────────────────
    cache_path = output_path.parent / "isrc_cache.json"
    isrc_cache: dict[str, Optional[str]] = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            isrc_cache = json.load(f)
        logger.info(f"ISRC-Cache: {len(isrc_cache)} Einträge geladen")

    # ── Phase 1: Kworb scrapen ───────────────────────────────────────────────
    logger.info(f"Phase 1: Kworb _weekly_totals scrapen ({len(args.markets)} Märkte)...")
    all_market_tracks: dict[str, list[dict]] = {}
    for market in args.markets:
        all_market_tracks[market] = scrape_market(market, args.min_streams, args.quality_only)

    aggregated = aggregate_markets(all_market_tracks, max_tracks=args.max_tracks)
    total = len(aggregated)
    market_counts = {m: len(t) for m, t in all_market_tracks.items()}

    if args.dry_run:
        hits = mids = 0
        for t in aggregated.values():
            _, label = compute_chart_score_and_label(t["chart_entries"])
            if label == "hit":
                hits += 1
            else:
                mids += 1

        print(f"\n{'='*60}")
        print(f"Dry-Run — Kworb _weekly_totals")
        print(f"  Märkte:         {', '.join(args.markets)}")
        print(f"  Min-Streams:    {args.min_streams:,}")
        print(f"  Tracks je Markt (vor Dedup):")
        for m, c in market_counts.items():
            print(f"    {m:6s}: {c:5d}")
        print(f"  Unique (dedup): {total}" + (f"  [Top {args.max_tracks} by streams]" if args.max_tracks else ""))
        print(f"  Labels:         {hits} Hits, {mids} Mids")
        print(f"  Output:         {output_path}")
        print(f"{'='*60}")
        return

    # ── Checkpoint + Resume-Setup ────────────────────────────────────────────
    checkpoint_path   = output_path.parent / "kworb_checkpoint.jsonl"
    miss_cache_path   = output_path.parent / "deezer_miss_cache.json"
    checkpoint_every  = args.checkpoint_every

    # Bestehende JSONL-Keys einlesen (bereits final gemergte Tracks überspringen)
    existing_keys: set[str] = set()
    for t in read_tracks(output_path):
        a = t.get("artist", "")
        ti = t.get("title", "")
        if a or ti:
            existing_keys.add(f"{a.lower()}|||{ti.lower()}")
    logger.info(f"Bestehende JSONL: {len(existing_keys)} Tracks")

    # Checkpoint-Datei laden (Tracks aus laufendem/abgebrochenem Run)
    accumulated_tracks: list[dict] = []
    checkpoint_keys: set[str] = set()
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    t = json.loads(line)
                    accumulated_tracks.append(t)
                    a  = t.get("artist", "")
                    ti = t.get("title", "")
                    checkpoint_keys.add(f"{a.lower()}|||{ti.lower()}")
        logger.info(f"Checkpoint geladen: {len(accumulated_tracks)} Tracks → Fortsetzen")

    # Miss-Cache laden (Tracks ohne Deezer-Match nicht erneut versuchen)
    miss_keys: set[str] = set()
    if miss_cache_path.exists():
        with open(miss_cache_path, "r", encoding="utf-8") as f:
            miss_keys = set(json.load(f))
        logger.info(f"Miss-Cache: {len(miss_keys)} Einträge")

    skip_keys = existing_keys | checkpoint_keys | miss_keys

    # ── Phase 2+3: MusicBrainz + Deezer ─────────────────────────────────────
    stats = {
        "hit": 0, "mid": 0,
        "mb_found": 0, "mb_miss": 0, "mb_cached": 0,
        "deezer_isrc": 0, "deezer_search": 0, "deezer_miss": 0,
        "skipped": 0,
    }

    track_list = list(aggregated.values())

    # Bereits verarbeitete Tracks überspringen
    todo_list = [
        t for t in track_list
        if f"{t['artist'].lower()}|||{t['title'].lower()}" not in skip_keys
    ]
    stats["skipped"] = len(track_list) - len(todo_list)
    if stats["skipped"]:
        logger.info(
            f"Resume: {stats['skipped']} Tracks übersprungen "
            f"({len(todo_list)} verbleibend)"
        )

    new_misses: list[str] = []

    for i, track in enumerate(todo_list, 1):
        artist        = track["artist"]
        title         = track["title"]
        chart_entries = track["chart_entries"]
        track_key     = f"{artist.lower()}|||{title.lower()}"

        if i % 50 == 0 or i == len(todo_list):
            logger.info(
                f"  [{i}/{len(todo_list)}] {artist[:30]} — {title[:30]}"
                f"  (MB-Cache: {len(isrc_cache)})"
            )

        chart_score, label = compute_chart_score_and_label(chart_entries)

        # MusicBrainz ISRC
        isrc = None
        if not args.skip_mb:
            was_cached = track_key in isrc_cache
            isrc = get_isrc_by_artist_title(artist, title, isrc_cache)
            if was_cached:
                stats["mb_cached"] += 1
            elif isrc:
                stats["mb_found"] += 1
            else:
                stats["mb_miss"] += 1

        # Deezer-Lookup
        deezer_data = None
        if isrc:
            deezer_data = lookup_deezer_by_isrc(isrc)
            if deezer_data:
                stats["deezer_isrc"] += 1

        if not deezer_data:
            deezer_data = lookup_deezer_by_search(artist, title)
            if deezer_data:
                stats["deezer_search"] += 1
            else:
                stats["deezer_miss"] += 1
                new_misses.append(track_key)
                logger.warning(f"Kein Deezer-Match: {artist} — {title}")

        if deezer_data:
            track_dict = build_track_dict(
                deezer_data, isrc, chart_entries, chart_score, label,
            )
            accumulated_tracks.append(track_dict)
            stats[label] += 1

        # ── Checkpoint ───────────────────────────────────────────────────────
        if i % checkpoint_every == 0:
            # Checkpoint-Datei (alle bisher gefundenen Tracks)
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                for t in accumulated_tracks:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")
            # Miss-Cache
            miss_keys.update(new_misses)
            new_misses = []
            with open(miss_cache_path, "w", encoding="utf-8") as f:
                json.dump(sorted(miss_keys), f, ensure_ascii=False)
            # ISRC-Cache
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(isrc_cache, f, ensure_ascii=False, indent=2)
            logger.info(
                f"  ✓ Checkpoint @ {i}/{len(todo_list)}: "
                f"{len(accumulated_tracks)} Tracks gespeichert"
            )

    # ISRC-Cache final sichern
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(isrc_cache, f, ensure_ascii=False, indent=2)
    logger.info(f"ISRC-Cache gespeichert: {len(isrc_cache)} Einträge → {cache_path}")

    # Miss-Cache final sichern
    miss_keys.update(new_misses)
    with open(miss_cache_path, "w", encoding="utf-8") as f:
        json.dump(sorted(miss_keys), f, ensure_ascii=False)
    logger.info(f"Miss-Cache gespeichert: {len(miss_keys)} Einträge → {miss_cache_path}")

    # ── Phase 4: Merge + schreiben ───────────────────────────────────────────
    updated, added = merge_tracks(output_path, accumulated_tracks)

    # Checkpoint-Datei aufräumen (Lauf erfolgreich abgeschlossen)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Checkpoint-Datei gelöscht (Lauf abgeschlossen)")

    # ── Zusammenfassung ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"scout_kworb.py — Abschluss")
    print(f"  Märkte:            {', '.join(args.markets)}")
    print(f"  Min-Streams:       {args.min_streams:,}")
    print(f"  Tracks je Markt (vor Dedup):")
    for m, c in market_counts.items():
        print(f"    {m:6s}: {c:5d}")
    print(f"  Unique (dedup):    {total}")
    if stats["skipped"]:
        print(f"  Übersprungen:      {stats['skipped']} (bereits verarbeitet)")
    if not args.skip_mb:
        print(f"  MusicBrainz:       {stats['mb_found']} neu / "
              f"{stats['mb_cached']} cached / {stats['mb_miss']} miss")
    print(f"  Deezer via ISRC:   {stats['deezer_isrc']}")
    print(f"  Deezer via Suche:  {stats['deezer_search']}")
    print(f"  Kein Deezer:       {stats['deezer_miss']}")
    print(f"  Labels:            {stats['hit']} Hits, {stats['mid']} Mids")
    print(f"  Output:            {updated} aktualisiert, {added} neu")
    print(f"  JSONL:             {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
