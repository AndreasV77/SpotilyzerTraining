"""
enrich_isrc.py
==============
Füllt fehlende ISRCs (isrc: null bzw. Feld fehlt komplett) in bestehenden
tracks.jsonl-Dateien nach — via MusicBrainz Artist+Title-Suche.

Hintergrund: main-Dataset (scout_deezer.py) hat nie ein isrc-Feld gesetzt.
spotify_charts und kworb wurden bisher mit --skip-mb gescoutet (schneller,
aber ohne ISRC). Dieses Skript holt das isoliert nach, ohne die Scouting-
Skripte selbst zu verlangsamen.

Design: für Mehrtage-Betrieb "nebenher" gedacht (MusicBrainz Rate-Limit
1 req/s → bei ~29.000 fehlenden ISRCs mehrere Stunden bis Tage Laufzeit).

  - Checkpointing: isrc_cache.json wird nach JEDEM Track gesichert (billig).
    Die eigentlichen tracks.jsonl-Dateien werden alle --checkpoint-every
    Tracks aktualisiert (teurer, da Full-Rewrite).
  - Resumable: beim Start wird pro Dataset neu ermittelt, welche Tracks noch
    kein isrc haben. Der Cache verhindert dabei doppelte MusicBrainz-Calls
    für bereits aufgelöste (Artist, Title)-Paare — auch nach hartem Abbruch
    (Ctrl+C, Fensterschließen, Stromausfall) geht dadurch höchstens der
    Zeitraum seit dem letzten jsonl-Checkpoint verloren, nie ein API-Call.
  - Unterbrechbar: Strg+C sichert sofort und beendet sauber. Einfach erneut
    starten, um fortzusetzen — kein Resume-Flag nötig.

Usage:
    python scripts/enrich_isrc.py                          # alle 3 Datasets
    python scripts/enrich_isrc.py --datasets kworb          # nur kworb
    python scripts/enrich_isrc.py --dry-run                 # nur Stats, keine Calls
    python scripts/enrich_isrc.py --limit 200                # Testlauf, dann stoppen
"""

import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import setup_logging, load_paths_config
from utils.metadata import read_tracks, update_tracks
from utils.musicbrainz import get_isrc_by_artist_title

DATASET_CHOICES = ["main", "spotify_charts", "kworb"]


def dataset_path(name: str, paths: dict) -> Path:
    """Löst den tracks.jsonl-Pfad für ein Dataset auf."""
    if name == "main":
        return paths["metadata"] / "tracks.jsonl"
    return paths["datasets"] / name / "tracks.jsonl"


def load_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache_path: Path, cache: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def flush_updates(dataset_paths: dict, pending: dict, logger) -> None:
    """Schreibt pending Updates pro Dataset zurück in die jeweilige tracks.jsonl."""
    for name, updates in pending.items():
        if not updates:
            continue
        update_tracks(dataset_paths[name], updates)
        logger.info(f"  ✓ Checkpoint {name}: {len(updates)} Tracks in tracks.jsonl geschrieben")
        updates.clear()


def main():
    parser = argparse.ArgumentParser(
        description="ISRC-Nachanreicherung via MusicBrainz für bestehende tracks.jsonl-Dateien"
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASET_CHOICES, default=DATASET_CHOICES,
        help=f"Welche Datasets bearbeiten (default: alle — {' '.join(DATASET_CHOICES)})",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=50, metavar="N",
        help="tracks.jsonl alle N neu aufgelöste Tracks aktualisieren (default: 50). "
             "isrc_cache.json wird davon unabhängig nach jedem Track gesichert.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Nach N neu aufgelösten (nicht gecachten) Tracks stoppen (default: alle)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Nur zeigen wie viele Tracks pro Dataset fehlende ISRCs haben, keine API-Calls",
    )
    args = parser.parse_args()

    logger = setup_logging("enrich_isrc")
    paths = load_paths_config()

    dataset_paths = {name: dataset_path(name, paths) for name in args.datasets}
    cache_path = paths["metadata"] / "isrc_cache.json"

    # ── Kandidaten sammeln ───────────────────────────────────────────────────
    todo: list[tuple[str, dict]] = []  # (dataset_name, track_dict)
    for name, path in dataset_paths.items():
        if not path.exists():
            logger.warning(f"{name}: {path} existiert nicht, übersprungen")
            continue
        tracks = read_tracks(path)
        missing = [t for t in tracks if t.get("isrc") is None and t.get("artist") and t.get("title")]
        logger.info(f"{name}: {len(tracks)} Tracks gesamt, {len(missing)} ohne ISRC")
        todo.extend((name, t) for t in missing)

    if args.dry_run:
        print("=" * 60)
        print("Dry-Run — ISRC-Enrichment")
        for name, path in dataset_paths.items():
            n = sum(1 for d, _ in todo if d == name)
            print(f"  {name:16s}: {n} Tracks ohne ISRC")
        print(f"  {'Gesamt':16s}: {len(todo)} Tracks")
        print("=" * 60)
        return

    if not todo:
        logger.info("Nichts zu tun — alle Tracks haben bereits ein ISRC.")
        return

    cache = load_cache(cache_path)
    logger.info(f"ISRC-Cache geladen: {len(cache)} Einträge")
    logger.info(f"{len(todo)} Tracks insgesamt ohne ISRC, Rate-Limit 1.1s/Call (nur bei Cache-Miss)")

    pending: dict[str, dict] = {name: {} for name in dataset_paths}
    stats = {"cached": 0, "found": 0, "miss": 0}
    new_lookups = 0
    start = time.time()

    def checkpoint():
        save_cache(cache_path, cache)
        flush_updates(dataset_paths, pending, logger)

    try:
        for i, (dataset_name, track) in enumerate(todo, 1):
            artist = track["artist"]
            title = track["title"]
            track_id = track["track_id"]
            cache_key = f"{artist.lower()}|||{title.lower()}"

            was_cached = cache_key in cache
            isrc = get_isrc_by_artist_title(artist, title, cache, logger=logger)

            if was_cached:
                stats["cached"] += 1
            else:
                new_lookups += 1
                if isrc:
                    stats["found"] += 1
                else:
                    stats["miss"] += 1

            pending[dataset_name][track_id] = {"isrc": isrc}

            # Cache nur bei tatsächlich neuem Eintrag sichern (billig, aber bei
            # reinen Cache-Hit-Serien unnötig oft schreiben vermeiden)
            if not was_cached:
                save_cache(cache_path, cache)

            if i % 10 == 0 or i == len(todo):
                elapsed = time.time() - start
                logger.info(
                    f"  [{i}/{len(todo)}] {artist[:30]} — {title[:30]}  "
                    f"(gefunden: {stats['found']}, miss: {stats['miss']}, "
                    f"cache: {stats['cached']}, {elapsed:.0f}s)"
                )

            total_pending = sum(len(u) for u in pending.values())
            if total_pending >= args.checkpoint_every:
                flush_updates(dataset_paths, pending, logger)

            if args.limit and new_lookups >= args.limit:
                logger.info(f"--limit {args.limit} erreicht, stoppe.")
                break

    except KeyboardInterrupt:
        logger.info("Unterbrochen (Strg+C) — sichere Fortschritt...")
    finally:
        checkpoint()

    logger.info(
        f"Fertig für diesen Lauf: {stats['found']} gefunden, {stats['miss']} nicht gefunden, "
        f"{stats['cached']} aus Cache. Einfach erneut starten, um fortzusetzen."
    )


if __name__ == "__main__":
    main()
