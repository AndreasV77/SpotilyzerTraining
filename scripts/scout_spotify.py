"""
scout_spotify.py
================
Liest Spotify Charts CSV-Dateien → ISRC via MusicBrainz → Deezer Track-ID.

Workflow:
  1. Alle regional-*-weekly-*.csv aus --input-Verzeichnis einlesen
  2. Tracks über alle Länder aggregieren (chart_entries pro Spotify-ID)
  3. ISRC via MusicBrainz (Spotify-ID als External Link), mit lokalem Cache
  4. Deezer-Lookup via ISRC (/track/isrc:{isrc}), Fallback: Artist+Title-Suche
  5. ChartScore + Label berechnen
  6. Merge in datasets/spotify_charts/tracks.jsonl

Output-Schema (pro Track):
  track_id       — Deezer Track-ID (Primärschlüssel)
  title, artist, album — Metadaten
  deezer_rank    — Deezer Popularity-Wert
  spotify_id     — Spotify Track-ID
  isrc           — ISRC (via MusicBrainz, oder null)
  chart_entries  — [{"country": "us", "rank": 1, "peak_rank": 1,
                     "weeks_on_chart": 1, "streams": 12796916}, ...]
  chart_score    — max(MarketWeight * ChartScore) über alle Märkte
  label          — hit/mid
  robustness     — "validated"
  dataset        — "spotify_charts"
  clusters       — ["charts_us", "charts_gb", ...]

ChartScore-Formel:
  ChartScore = (1 / peak_rank) * log(weeks_on_chart + 1)
  HitScore   = max(MarketWeight[country] * ChartScore)

Hit-Kriterien:
  Tier 1 (weight=1.0):  peak_rank ≤ 100  (US, UK/GB, Global)
  Tier 2 (weight=0.85): peak_rank ≤ 50   (DE, JP, BR, FR, AU, CA)
  Tier 3 (weight=0.70): peak_rank ≤ 20   (MX, ES, IT)
"""

import sys
import time
import csv
import json
import math
import argparse
import re
from pathlib import Path
from typing import Optional

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import setup_logging, load_paths_config, ensure_dir
from utils.metadata import read_tracks_as_dict, merge_tracks

# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DEEZER_API_BASE = "https://api.deezer.com"
MB_API_BASE     = "https://musicbrainz.org/ws/2"
MB_USER_AGENT   = "SpotilyzerTraining/1.0 (github.com/AndreasV77/SpotilyzerTraining)"

# Markt-Gewichte für ChartScore
MARKET_WEIGHTS = {
    "global": 1.0,
    "us":     1.0,
    "gb":     1.0,   # UK
    "de":     0.85,
    "jp":     0.85,
    "br":     0.85,
    "fr":     0.85,
    "au":     0.85,
    "ca":     0.85,
    "mx":     0.70,
    "es":     0.70,
    "it":     0.70,
}

# Hit-Schwellenwerte (peak_rank) je Tier
HIT_THRESHOLDS = {
    1.0:  100,   # Tier 1: Top 100
    0.85:  50,   # Tier 2: Top 50
    0.70:  20,   # Tier 3: Top 20
}

logger = None


# ══════════════════════════════════════════════════════════════════════════════
# CSV PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_country_from_filename(path: Path) -> Optional[str]:
    """Extrahiert Länder-Code aus CSV-Dateiname: regional-{country}-weekly-*.csv"""
    m = re.match(r"regional-([a-z]+)-(?:daily|weekly)-", path.name)
    return m.group(1) if m else None


def load_spotify_csvs(input_dir: Path) -> dict[str, dict]:
    """
    Lädt alle regional-*-weekly-*.csv aus input_dir.

    Returns:
        {spotify_id: {"spotify_id": ..., "artist": ..., "title": ...,
                      "chart_entries": [...]}}
    """
    tracks: dict[str, dict] = {}
    files_loaded = 0

    for csv_path in sorted(input_dir.glob("regional-*-weekly-*.csv")):
        country = parse_country_from_filename(csv_path)
        if not country:
            continue

        with open(csv_path, newline="", encoding="utf-8-sig") as f:  # utf-8-sig strips BOM
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                spotify_id = row["uri"].split(":")[-1]

                if spotify_id not in tracks:
                    tracks[spotify_id] = {
                        "spotify_id": spotify_id,
                        "artist": row["artist_names"],
                        "title":  row["track_name"],
                        "chart_entries": [],
                    }

                tracks[spotify_id]["chart_entries"].append({
                    "country":       country,
                    "rank":          int(row["rank"]),
                    "peak_rank":     int(row["peak_rank"]),
                    "weeks_on_chart": int(row["weeks_on_chart"]),
                    "streams":       int(row["streams"]),
                })
                count += 1

        logger.info(f"  {csv_path.name}: {count} Tracks ({country})")
        files_loaded += 1

    logger.info(f"{files_loaded} CSV-Dateien geladen, {len(tracks)} unique Tracks")
    return tracks


# ══════════════════════════════════════════════════════════════════════════════
# CHART SCORE & LABEL
# ══════════════════════════════════════════════════════════════════════════════

def compute_chart_score_and_label(chart_entries: list[dict]) -> tuple[float, str]:
    """
    Berechnet HitScore und Label aus Chart-Einträgen.

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
        time.sleep(1.1)  # MusicBrainz: max 1 req/s
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


def get_isrc(spotify_id: str, cache: dict) -> Optional[str]:
    """
    Holt ISRC für eine Spotify Track-ID via MusicBrainz.
    Nutzt lokalen Cache (spotify_id → isrc) für Wiederholungen.

    Zwei API-Calls:
      1. /url?resource=https://open.spotify.com/track/{id} → MBID
      2. /recording/{mbid}?inc=isrcs → ISRC
    """
    if spotify_id in cache:
        return cache[spotify_id]

    # Schritt 1: Spotify-ID → MusicBrainz Recording-ID
    data = mb_get("url", params={
        "resource": f"https://open.spotify.com/track/{spotify_id}",
        "inc":      "recording-rels",
        "fmt":      "json",
    })

    mbid = None
    if data:
        for rel in data.get("relations", []):
            if rel.get("recording"):
                mbid = rel["recording"]["id"]
                break

    if not mbid:
        cache[spotify_id] = None
        return None

    # Schritt 2: MBID → ISRC
    data = mb_get(f"recording/{mbid}", params={"inc": "isrcs", "fmt": "json"})

    isrc = None
    if data:
        isrcs = data.get("isrcs", [])
        if isrcs:
            isrc = isrcs[0]

    cache[spotify_id] = isrc
    return isrc


# ══════════════════════════════════════════════════════════════════════════════
# DEEZER
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
    """
    Deezer-Track-Suche via Artist+Title (Fallback).
    Bei mehreren Artists (kommasepariert) wird nur der erste verwendet.
    """
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
    spotify_id: str,
    isrc: Optional[str],
    chart_entries: list[dict],
    chart_score: float,
    label: str,
) -> dict:
    """Baut Track-Dict im SpotilyzerTraining-Format."""
    countries = sorted({e["country"] for e in chart_entries})
    return {
        "track_id":     deezer_data["id"],
        "title":        deezer_data.get("title", ""),
        "artist":       deezer_data.get("artist", {}).get("name", ""),
        "album":        deezer_data.get("album", {}).get("title", ""),
        "deezer_rank":  deezer_data.get("rank", 0),
        "spotify_id":   spotify_id,
        "isrc":         isrc,
        "chart_entries": chart_entries,
        "chart_score":  chart_score,
        "label":        label,
        "robustness":   "validated",
        "dataset":      "spotify_charts",
        "clusters":     [f"charts_{c}" for c in countries],
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global logger

    parser = argparse.ArgumentParser(
        description="Spotify Charts CSV → ISRC (MusicBrainz) → Deezer Track-IDs"
    )
    parser.add_argument(
        "--input", type=Path,
        help="Verzeichnis mit Spotify CSV-Dateien (default: neuestes in spotify_charts/)",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Output JSONL (default: datasets/spotify_charts/tracks.jsonl)",
    )
    parser.add_argument(
        "--skip-mb", action="store_true",
        help="MusicBrainz überspringen, direkt Deezer-Suche (schneller, kein ISRC)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="CSV einlesen + Stats zeigen, keine API-Calls",
    )
    args = parser.parse_args()

    logger = setup_logging("scout_spotify")
    paths  = load_paths_config()

    # ── Input-Verzeichnis ────────────────────────────────────────────────────
    if args.input:
        input_dir = args.input
    else:
        spotify_base = paths.get("spotify_charts", paths["data_root"] / "spotify")
        subdirs = sorted([d for d in spotify_base.iterdir() if d.is_dir()])
        if not subdirs:
            logger.error(f"Keine Unterverzeichnisse in {spotify_base}")
            sys.exit(1)
        input_dir = subdirs[-1]

    logger.info(f"Input: {input_dir}")

    # ── Output-Pfad ──────────────────────────────────────────────────────────
    if args.output:
        output_path = args.output
    else:
        datasets_dir = paths.get("datasets", paths["data_root"] / "datasets")
        output_path  = datasets_dir / "spotify_charts" / "tracks.jsonl"

    ensure_dir(output_path.parent)
    logger.info(f"Output: {output_path}")

    # ── ISRC-Cache ───────────────────────────────────────────────────────────
    cache_path = output_path.parent / "isrc_cache.json"
    isrc_cache: dict[str, Optional[str]] = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            isrc_cache = json.load(f)
        logger.info(f"ISRC-Cache: {len(isrc_cache)} Einträge geladen")

    # ── Phase 1: CSV einlesen ────────────────────────────────────────────────
    logger.info("Phase 1: CSV-Dateien einlesen...")
    tracks_data = load_spotify_csvs(input_dir)
    total = len(tracks_data)

    if args.dry_run:
        countries = set()
        hits = mids = 0
        for t in tracks_data.values():
            countries.update(e["country"] for e in t["chart_entries"])
            _, label = compute_chart_score_and_label(t["chart_entries"])
            if label == "hit":
                hits += 1
            else:
                mids += 1
        print(f"\nDry-Run — {input_dir.name}")
        print(f"  Unique Tracks: {total}")
        print(f"  Märkte:        {sorted(countries)}")
        print(f"  Geschätzte Labels: {hits} Hits, {mids} Mids")
        return

    # ── Phase 2+3: MusicBrainz + Deezer ─────────────────────────────────────
    stats = {
        "hit": 0, "mid": 0,
        "mb_found": 0, "mb_miss": 0, "mb_cached": 0,
        "deezer_isrc": 0, "deezer_search": 0, "deezer_miss": 0,
    }
    new_tracks = []
    tracks_list = list(tracks_data.values())

    for i, track in enumerate(tracks_list, 1):
        spotify_id = track["spotify_id"]
        artist     = track["artist"]
        title      = track["title"]

        if i % 25 == 0 or i == total:
            logger.info(
                f"  [{i}/{total}] {artist[:30]} — {title[:30]}"
                f"  (ISRC-Cache: {len(isrc_cache)})"
            )

        chart_score, label = compute_chart_score_and_label(track["chart_entries"])

        # MusicBrainz ISRC
        isrc = None
        if not args.skip_mb:
            was_cached = spotify_id in isrc_cache
            isrc = get_isrc(spotify_id, isrc_cache)
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
                logger.warning(f"Kein Deezer-Match: {artist} — {title}")
                continue

        new_tracks.append(build_track_dict(
            deezer_data, spotify_id, isrc,
            track["chart_entries"], chart_score, label,
        ))
        stats[label] += 1

    # Cache sichern
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(isrc_cache, f, ensure_ascii=False, indent=2)
    logger.info(f"ISRC-Cache gespeichert: {len(isrc_cache)} Einträge → {cache_path}")

    # ── Phase 4: Merge + schreiben ───────────────────────────────────────────
    updated, added = merge_tracks(output_path, new_tracks)

    # ── Zusammenfassung ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"scout_spotify.py — Abschluss")
    print(f"  Input:             {input_dir.name}")
    print(f"  Unique Tracks:     {total}")
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
