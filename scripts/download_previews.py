"""
download_previews.py
====================
Laedt die 30s-Preview-MP3s von Deezer herunter.

WICHTIG: Deezer Preview-URLs sind nur ~15 Minuten gueltig!
Daher holt dieses Script die URLs frisch vor jedem Download.

Input: metadata/tracks.jsonl (aus scout_deezer.py)
Output:
  - previews/{shard}/{track_id}.mp3 (MD5-Sharding, mit ID3-Tags)
  - metadata/tracks.jsonl (file_path-Feld wird hinzugefuegt)
"""

import sys
import re
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

from _utils import (
    setup_logging,
    load_paths_config,
    ensure_dir,
)
from utils.paths import get_preview_path, get_relative_preview_path, ensure_shard_dir
from utils.metadata import (
    read_tracks,
    update_tracks,
    filter_tracks,
    get_tracks_jsonl_path,
)

# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DEEZER_API = "https://api.deezer.com"
REQUEST_TIMEOUT = 15
MAX_WORKERS = 4
RETRY_COUNT = 2
API_DELAY = 0.1

# Logger (wird in main() initialisiert)
logger = None


# ══════════════════════════════════════════════════════════════════════════════
# ID3-TAGGING
# ══════════════════════════════════════════════════════════════════════════════

def tag_mp3(filepath: Path, track: dict):
    """
    Setzt ID3-Tags auf eine MP3-Datei.

    Tags:
      TIT2 — Title
      TPE1 — Artist
      TALB — Album
      COMM — Comment: deezer:{track_id}|clusters:{cluster1,cluster2}
    """
    try:
        from mutagen.id3 import ID3, TIT2, TPE1, TALB, COMM
        from mutagen.id3 import ID3NoHeaderError
    except ImportError:
        if logger:
            logger.warning("mutagen nicht installiert, ID3-Tagging uebersprungen")
        return

    try:
        try:
            tags = ID3(filepath)
        except ID3NoHeaderError:
            tags = ID3()

        tags.delall("TIT2")
        tags.delall("TPE1")
        tags.delall("TALB")
        tags.delall("COMM")

        tags.add(TIT2(encoding=3, text=track.get("title", "")))
        tags.add(TPE1(encoding=3, text=track.get("artist", "")))
        tags.add(TALB(encoding=3, text=track.get("album", "")))

        clusters = ",".join(track.get("clusters", []))
        comment = f"deezer:{track['track_id']}|clusters:{clusters}"
        tags.add(COMM(encoding=3, lang="eng", desc="", text=comment))

        tags.save(filepath)

    except Exception as e:
        if logger:
            logger.warning(f"ID3-Tagging fehlgeschlagen fuer {filepath.name}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# API-HELFER
# ══════════════════════════════════════════════════════════════════════════════

def extract_expiry_from_url(url: str) -> int | None:
    """Extrahiert den Expiry-Timestamp aus einer Deezer Preview-URL."""
    if not url:
        return None
    match = re.search(r'exp=(\d+)', url)
    if match:
        return int(match.group(1))
    return None


def is_url_expired(url: str, buffer_seconds: int = 60) -> bool:
    """Prueft ob eine Preview-URL abgelaufen ist."""
    expiry = extract_expiry_from_url(url)
    if expiry is None:
        return False
    now = int(time.time())
    return now >= (expiry - buffer_seconds)


def get_fresh_preview_url(track_id: int) -> tuple[str | None, int | None]:
    """Holt eine frische Preview-URL fuer einen Track."""
    try:
        response = requests.get(
            f"{DEEZER_API}/track/{track_id}",
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            url = data.get("preview")
            expiry = extract_expiry_from_url(url) if url else None
            return url, expiry
        return None, None
    except Exception:
        return None, None


# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD-LOGIK
# ══════════════════════════════════════════════════════════════════════════════

def download_preview(
    track: dict,
    previews_dir: Path,
) -> tuple[int, bool, str]:
    """
    Holt frische URL und laedt Preview herunter.
    Speichert in MD5-Shard-Verzeichnis mit ID3-Tags.

    Returns:
        (track_id, success, message)
    """
    track_id = track["track_id"]
    filepath = get_preview_path(track_id, previews_dir)

    # Skip wenn bereits vorhanden und gross genug
    if filepath.exists() and filepath.stat().st_size > 50000:
        return (track_id, True, "skipped")

    # Shard-Verzeichnis erstellen
    ensure_shard_dir(track_id, previews_dir)

    # Frische URL holen
    time.sleep(API_DELAY)
    preview_url, expiry = get_fresh_preview_url(track_id)

    if not preview_url:
        return (track_id, False, "no preview url")

    if is_url_expired(preview_url):
        return (track_id, False, "url expired immediately")

    # Download
    for attempt in range(RETRY_COUNT + 1):
        try:
            response = requests.get(preview_url, timeout=REQUEST_TIMEOUT, stream=True)

            if response.status_code == 200:
                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                if filepath.stat().st_size < 10000:
                    filepath.unlink()
                    return (track_id, False, "file too small")

                # ID3-Tags setzen
                tag_mp3(filepath, track)

                return (track_id, True, "downloaded")

            elif response.status_code == 403:
                return (track_id, False, "403 forbidden")
            elif response.status_code == 404:
                return (track_id, False, "404 not found")
            else:
                if attempt < RETRY_COUNT:
                    time.sleep(1)
                    continue
                return (track_id, False, f"HTTP {response.status_code}")

        except requests.exceptions.Timeout:
            if attempt < RETRY_COUNT:
                time.sleep(1)
                continue
            return (track_id, False, "timeout")
        except requests.exceptions.RequestException as e:
            if attempt < RETRY_COUNT:
                time.sleep(1)
                continue
            return (track_id, False, str(e)[:50])

    return (track_id, False, "max retries")


def download_batch(
    tracks: list[dict],
    previews_dir: Path,
    max_workers: int,
) -> tuple[dict, list[int]]:
    """
    Laedt eine Batch von Previews herunter.

    Returns:
        (stats_dict, list_of_successful_track_ids)
    """
    stats = {"success": 0, "failed": 0, "skipped": 0, "errors": []}
    successful_ids = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_preview, track, previews_dir): track
            for track in tracks
        }

        with tqdm(total=len(futures), desc="Downloading", unit="file") as pbar:
            for future in as_completed(futures):
                track_id, success, message = future.result()

                if success:
                    if message == "skipped":
                        stats["skipped"] += 1
                    else:
                        stats["success"] += 1
                    successful_ids.append(track_id)
                else:
                    stats["failed"] += 1
                    stats["errors"].append((track_id, message))

                    if logger and "expired" in message:
                        logger.warning(f"URL expired fuer Track {track_id}")

                pbar.update(1)

    return stats, successful_ids


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global logger

    paths = load_paths_config()

    previews_dir = paths.get("previews")
    metadata_dir = paths.get("metadata")
    jsonl_path = get_tracks_jsonl_path(metadata_dir)

    parser = argparse.ArgumentParser(
        description="Download Deezer Preview-MP3s (MD5-Sharding, ID3-Tags)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Parallele Downloads (default: {MAX_WORKERS})"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximale Anzahl Downloads (0 = alle)"
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default=None,
        help="Nur bestimmten Cluster downloaden"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Zeige was heruntergeladen wuerde, ohne tatsaechlich zu laden"
    )
    args = parser.parse_args()

    # Logging
    logger = setup_logging("download", log_dir=paths.get("logs"))

    # Tracks aus JSONL laden
    if not jsonl_path.exists():
        print(f"Fehler: tracks.jsonl nicht gefunden: {jsonl_path}")
        print(f"  -> Erst 'python scripts/scout_deezer.py' ausfuehren!")
        logger.error(f"tracks.jsonl nicht gefunden: {jsonl_path}")
        sys.exit(1)

    all_tracks = read_tracks(jsonl_path)

    print(f"{'=' * 79}")
    print(f"  DEEZER PREVIEW DOWNLOADER")
    print(f"  (MD5-Sharding, ID3-Tags, frische URLs)")
    print(f"{'=' * 79}")
    print(f"  JSONL:   {jsonl_path}")
    print(f"  Tracks:  {len(all_tracks)}")

    # Filter nach Cluster
    if args.cluster:
        all_tracks = filter_tracks(all_tracks, cluster=args.cluster)
        print(f"  Filter:  cluster={args.cluster} -> {len(all_tracks)} Tracks")

    # Limit
    if args.limit > 0:
        all_tracks = all_tracks[:args.limit]
        print(f"  Limit:   {args.limit} Tracks")

    # Cluster-Verteilung
    cluster_counts = {}
    for t in all_tracks:
        for c in t.get("clusters", []):
            cluster_counts[c] = cluster_counts.get(c, 0) + 1
    print(f"\n  Cluster-Verteilung:")
    for cluster, count in sorted(cluster_counts.items()):
        print(f"    {cluster:25} {count:>5}")

    if args.dry_run:
        print(f"\n  [DRY RUN] Keine Downloads durchgefuehrt.")
        return

    # Output-Verzeichnis
    ensure_dir(previews_dir)
    print(f"\n  Output:  {previews_dir}")

    logger.info(
        f"Download gestartet: {len(all_tracks)} Tracks, {args.workers} Workers, "
        f"Output: {previews_dir}"
    )

    # Geschaetzte Zeit
    est_time = len(all_tracks) * (API_DELAY + 0.5) / args.workers / 60
    print(f"  Geschaetzte Zeit: ~{est_time:.0f} Minuten bei {args.workers} Workern")
    print(f"\n  Starte Downloads...\n")

    stats, successful_ids = download_batch(all_tracks, previews_dir, args.workers)

    # file_path in JSONL aktualisieren
    if successful_ids:
        file_path_updates = {}
        for tid in successful_ids:
            file_path_updates[tid] = {"file_path": get_relative_preview_path(tid)}

        update_tracks(jsonl_path, file_path_updates)
        print(f"\n  JSONL aktualisiert: {len(successful_ids)} Tracks mit file_path")

    # Ergebnis
    print(f"\n{'─' * 79}")
    print(f"  DOWNLOAD COMPLETE")
    print(f"{'─' * 79}")
    print(f"  Erfolgreich:    {stats['success']:>5}")
    print(f"  Uebersprungen:  {stats['skipped']:>5}  (bereits vorhanden)")
    print(f"  Fehlgeschlagen: {stats['failed']:>5}")

    if stats["errors"]:
        error_types = {}
        for track_id, message in stats["errors"]:
            error_types[message] = error_types.get(message, 0) + 1

        print(f"\n  Fehler-Zusammenfassung:")
        for error, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"    {error:30} {count:>5}")

    # Speicherplatz-Info (traversiere Shard-Verzeichnisse)
    total_size = 0
    file_count = 0
    for mp3 in previews_dir.rglob("*.mp3"):
        total_size += mp3.stat().st_size
        file_count += 1

    print(f"\n  Speicherplatz: {total_size / (1024*1024):.1f} MB")
    print(f"  Dateien:       {file_count}")

    if file_count > 0:
        print(f"  Durchschnitt:  {total_size / file_count / 1024:.1f} KB pro Datei")

    logger.info(
        f"Download abgeschlossen: {stats['success']} ok, "
        f"{stats['skipped']} skipped, {stats['failed']} failed"
    )


if __name__ == "__main__":
    main()
