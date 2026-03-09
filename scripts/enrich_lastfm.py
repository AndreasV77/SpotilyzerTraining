"""
enrich_lastfm.py
================
Reichert Tracks in tracks.jsonl mit Last.fm-Metriken an.

Workflow:
1. Liest metadata/tracks.jsonl
2. Sucht jeden Track auf Last.fm via Artist + Title
3. Fuzzy-Matching fuer robuste Zuordnung
4. Schreibt lastfm_playcount, lastfm_listeners, lastfm_tags zurueck in JSONL

Voraussetzungen:
- Last.fm API-Key in .env (LASTFM_API_KEY)
- pip install pylast rapidfuzz python-dotenv tqdm
"""

import os
import sys
import re
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from tqdm import tqdm

try:
    import pylast
except ImportError:
    print("pylast nicht installiert. Bitte: pip install pylast")
    sys.exit(1)

try:
    from rapidfuzz import fuzz
except ImportError:
    print("rapidfuzz nicht installiert. Bitte: pip install rapidfuzz")
    sys.exit(1)

from _utils import (
    setup_logging,
    load_paths_config,
    ensure_dir,
)
from utils.metadata import (
    read_tracks,
    update_tracks,
    get_tracks_jsonl_path,
)

# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

REQUEST_DELAY = 0.25  # Sekunden zwischen API-Calls
MATCH_THRESHOLD = 85  # Minimum Fuzzy-Score fuer Match (0-100)
MAX_RETRIES = 3

# Logger (wird in main() initialisiert)
logger = None


# ══════════════════════════════════════════════════════════════════════════════
# HILFSFUNKTIONEN
# ══════════════════════════════════════════════════════════════════════════════

def normalize_string(s: str) -> str:
    """Normalisiert String fuer Fuzzy-Matching."""
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"\s*[\(\[].*?[\)\]]", "", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"^the\s+", "", s)
    s = re.sub(r"\s+(live|remix|remaster(ed)?|edit|version|radio edit).*$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compute_match_confidence(
    deezer_artist: str,
    deezer_title: str,
    lastfm_artist: str,
    lastfm_title: str,
) -> float:
    """Berechnet Match-Konfidenz (0.0 - 1.0)."""
    artist_score = fuzz.ratio(
        normalize_string(deezer_artist),
        normalize_string(lastfm_artist),
    )
    title_score = fuzz.ratio(
        normalize_string(deezer_title),
        normalize_string(lastfm_title),
    )
    combined = 0.4 * artist_score + 0.6 * title_score
    return combined / 100.0


@dataclass
class LastFMResult:
    """Ergebnis einer Last.fm-Abfrage."""
    playcount: Optional[int] = None
    listeners: Optional[int] = None
    tags: list = None
    matched: bool = False
    match_confidence: float = 0.0
    error: Optional[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


def fetch_lastfm_data(
    network: pylast.LastFMNetwork,
    artist: str,
    title: str,
) -> LastFMResult:
    """Holt Track-Daten von Last.fm."""
    result = LastFMResult()

    for attempt in range(MAX_RETRIES):
        try:
            track = network.get_track(artist, title)

            playcount = track.get_playcount()
            listeners = track.get_listener_count()

            corrected_artist = track.get_artist().get_name()
            corrected_title = track.get_title()

            confidence = compute_match_confidence(
                artist, title,
                corrected_artist, corrected_title,
            )

            result.playcount = playcount
            result.listeners = listeners
            result.matched = confidence >= (MATCH_THRESHOLD / 100.0)
            result.match_confidence = confidence

            # Tags holen
            try:
                top_tags = track.get_top_tags(limit=5)
                result.tags = [tag.item.get_name().lower() for tag in top_tags]
            except Exception:
                result.tags = []

            return result

        except pylast.WSError as e:
            if "Track not found" in str(e):
                result.error = "not_found"
                return result
            elif "Rate limit" in str(e).lower():
                time.sleep(5)
                continue
            else:
                result.error = str(e)
                return result

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
                continue
            result.error = str(e)
            return result

    return result


# ══════════════════════════════════════════════════════════════════════════════
# HAUPTLOGIK
# ══════════════════════════════════════════════════════════════════════════════

def enrich_tracks(
    jsonl_path: Path,
    api_key: str,
    api_secret: str = "",
    resume: bool = True,
) -> dict:
    """
    Reichert Tracks in tracks.jsonl mit Last.fm-Metriken an.

    Returns:
        Dict mit Statistiken
    """
    network = pylast.LastFMNetwork(
        api_key=api_key,
        api_secret=api_secret if api_secret else None,
    )

    tracks = read_tracks(jsonl_path)
    print(f"  Geladen: {len(tracks)} Tracks aus {jsonl_path}")

    stats = {
        "total": len(tracks),
        "processed": 0,
        "matched": 0,
        "not_found": 0,
        "low_confidence": 0,
        "errors": 0,
        "skipped": 0,
    }

    updates = {}
    save_interval = 100

    for track in tqdm(tracks, desc="Last.fm Enrichment"):
        track_id = track["track_id"]
        artist = track.get("artist", "")
        title = track.get("title", "")

        # Resume: Skip wenn bereits mit Last.fm-Daten vorhanden
        if resume and track.get("lastfm_playcount") is not None:
            stats["skipped"] += 1
            continue

        if not artist or not title:
            stats["errors"] += 1
            if logger:
                logger.warning(f"Track {track_id}: kein Artist/Title")
            continue

        result = fetch_lastfm_data(network, artist, title)
        time.sleep(REQUEST_DELAY)

        stats["processed"] += 1

        if result.error == "not_found":
            stats["not_found"] += 1
            if logger:
                logger.info(f"Nicht gefunden: {artist} - {title}")
            continue
        elif result.error:
            stats["errors"] += 1
            if logger:
                logger.warning(f"Fehler bei {artist} - {title}: {result.error}")
            continue

        if not result.matched:
            stats["low_confidence"] += 1
            if logger:
                logger.info(
                    f"Low confidence ({result.match_confidence:.2f}): "
                    f"{artist} - {title}"
                )
            continue

        stats["matched"] += 1

        updates[track_id] = {
            "lastfm_playcount": result.playcount,
            "lastfm_listeners": result.listeners,
            "lastfm_tags": result.tags,
        }

        # Zwischenspeichern
        if len(updates) >= save_interval:
            update_tracks(jsonl_path, updates)
            tqdm.write(f"  Zwischenstand gespeichert ({stats['processed']} verarbeitet)")
            updates = {}

    # Rest speichern
    if updates:
        update_tracks(jsonl_path, updates)

    return stats


def print_stats(stats: dict):
    """Gibt Statistiken formatiert aus."""
    processed = max(1, stats["processed"])
    print(f"""
{'=' * 79}
  LAST.FM ENRICHMENT COMPLETE
{'=' * 79}
  Tracks gesamt:        {stats['total']:>6}
  Bereits verarbeitet:  {stats['skipped']:>6}
  Neu verarbeitet:      {stats['processed']:>6}
{'─' * 79}
  Gematcht:             {stats['matched']:>6}  ({100*stats['matched']/processed:.1f}%)
  Nicht gefunden:       {stats['not_found']:>6}  ({100*stats['not_found']/processed:.1f}%)
  Low Confidence:       {stats['low_confidence']:>6}  ({100*stats['low_confidence']/processed:.1f}%)
  Fehler:               {stats['errors']:>6}
{'=' * 79}
""")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global logger

    paths = load_paths_config()
    metadata_dir = paths.get("metadata")
    jsonl_path = get_tracks_jsonl_path(metadata_dir)

    parser = argparse.ArgumentParser(
        description="Reichert Tracks mit Last.fm-Metriken an (JSONL)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Nicht fortsetzen, alle Tracks neu verarbeiten",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Pfad zur .env-Datei mit API-Keys",
    )
    args = parser.parse_args()

    # Logging
    logger = setup_logging("enrichment", log_dir=paths.get("logs"))

    # .env laden
    load_dotenv(args.env_file)

    api_key = os.getenv("LASTFM_API_KEY")
    api_secret = os.getenv("LASTFM_API_SECRET", "")

    if not api_key:
        print("Fehler: LASTFM_API_KEY nicht in .env gefunden!")
        print("  API-Key erstellen: https://www.last.fm/api/account/create")
        sys.exit(1)

    if not jsonl_path.exists():
        print(f"Fehler: tracks.jsonl nicht gefunden: {jsonl_path}")
        print("  -> Erst 'python scripts/scout_deezer.py' ausfuehren!")
        sys.exit(1)

    print(f"{'=' * 79}")
    print(f"  LAST.FM ENRICHMENT")
    print(f"{'=' * 79}")
    print(f"  JSONL:   {jsonl_path}")
    print(f"  Resume:  {'Nein' if args.no_resume else 'Ja'}")

    logger.info(f"Enrichment gestartet: {jsonl_path}")

    stats = enrich_tracks(
        jsonl_path=jsonl_path,
        api_key=api_key,
        api_secret=api_secret,
        resume=not args.no_resume,
    )

    print_stats(stats)

    logger.info(
        f"Enrichment abgeschlossen: {stats['matched']} gematcht, "
        f"{stats['not_found']} nicht gefunden, {stats['errors']} Fehler"
    )


if __name__ == "__main__":
    main()
