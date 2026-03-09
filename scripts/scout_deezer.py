"""
scout_deezer.py
===============
Scouting-Script für die MERT-Embedding-Architektur.
Nutzt die Deezer API (kostenlos, ohne Auth) statt Spotify.

Workflow:
1. Für jeden Cluster: Seed-Artists suchen → Top-Tracks holen
2. Optional: Deezer-Radios des Clusters abrufen (/radio/{id}/tracks)
3. preview_url und rank erfassen
4. Ergebnis in metadata/tracks.jsonl mergen

Cluster-Definitionen in: configs/clusters.yaml
Radio-IDs je Cluster im Feld "radios" der Cluster-Definition.

Output:
  metadata/tracks.jsonl (JSONL, eine Zeile pro Track)
"""

import sys
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import requests

from _utils import (
    setup_logging,
    load_paths_config,
    load_thresholds_config,
    load_clusters_config,
    get_genre_clusters,
    get_charts_config,
    get_scouting_config,
    ensure_dir,
)
from utils.metadata import merge_tracks, read_tracks, get_tracks_jsonl_path

# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DEEZER_API_BASE = "https://api.deezer.com"

# Logger (wird in main() initialisiert)
logger = None


# ══════════════════════════════════════════════════════════════════════════════
# DATENSTRUKTUREN
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClusterStats:
    cluster_id: str
    display_name: str
    total_tracks: int = 0
    unique_tracks: int = 0
    hits: int = 0
    mids: int = 0
    flops: int = 0
    seed_artists_found: int = 0
    seed_artists_not_found: list = field(default_factory=list)
    radios_scouted: int = 0
    radio_tracks_added: int = 0
    avg_rank: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# API-HELFER
# ══════════════════════════════════════════════════════════════════════════════

def api_get(
    endpoint: str,
    params: dict = None,
    request_delay: float = 0.25,
    max_retries: int = 3,
) -> Optional[dict]:
    """Führt einen GET-Request gegen die Deezer API aus."""
    url = f"{DEEZER_API_BASE}/{endpoint}"

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            time.sleep(request_delay)

            if response.status_code == 200:
                data = response.json()
                if "error" in data:
                    msg = data["error"].get("message", "Unknown error")
                    if logger:
                        logger.warning(f"API Error: {msg} ({url})")
                    return None
                return data
            elif response.status_code == 429:
                if logger:
                    logger.warning("Rate limited, waiting 5s...")
                time.sleep(5)
                continue
            else:
                if logger:
                    logger.warning(f"HTTP {response.status_code}: {url}")
                return None

        except requests.exceptions.RequestException as e:
            if logger:
                logger.error(f"Request Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None

    return None


def search_artist(artist_name: str, **api_kwargs) -> Optional[dict]:
    """Sucht einen Artist nach Namen und gibt den besten Match zurück."""
    data = api_get("search/artist", {"q": artist_name, "limit": 5}, **api_kwargs)
    if not data or not data.get("data"):
        return None

    for artist in data["data"]:
        if artist["name"].lower() == artist_name.lower():
            return artist

    return data["data"][0]


def get_artist_top_tracks(artist_id: int, limit: int = 25, **api_kwargs) -> list[dict]:
    """Holt die Top-Tracks eines Artists."""
    data = api_get(f"artist/{artist_id}/top", {"limit": limit}, **api_kwargs)
    if not data or not data.get("data"):
        return []
    return data["data"]


def get_playlist_tracks(playlist_id: int, limit: int = 100, **api_kwargs) -> list[dict]:
    """Holt Tracks aus einer Playlist (z.B. Country Charts)."""
    data = api_get(f"playlist/{playlist_id}/tracks", {"limit": limit}, **api_kwargs)
    if not data or not data.get("data"):
        return []
    return data["data"]


def get_radio_tracks(radio_id: int, **api_kwargs) -> list[dict]:
    """Holt Tracks aus einem Deezer-Radio (liefert ~40 Tracks)."""
    data = api_get(f"radio/{radio_id}/tracks", **api_kwargs)
    if not data or not data.get("data"):
        return []
    return data["data"]


# ══════════════════════════════════════════════════════════════════════════════
# TRACK-SAMMLUNG
# ══════════════════════════════════════════════════════════════════════════════

def extract_track_dict(track: dict, cluster: str) -> Optional[dict]:
    """
    Extrahiert relevante Felder aus Deezer-Track-Objekt als Dict.

    Erzeugt ein Dict im tracks.jsonl-Format.
    """
    if not track or not track.get("id"):
        return None

    artist = track.get("artist", {})

    return {
        "track_id": track["id"],
        "title": track.get("title", ""),
        "artist": artist.get("name", "Unknown"),
        "album": track.get("album", {}).get("title", ""),
        "clusters": [cluster],
        "deezer_rank": track.get("rank", 0),
    }


def collect_tracks_for_cluster(
    cluster_id: str,
    cluster_def: dict,
    tracks_per_artist: int,
    rank_thresholds: dict,
    api_kwargs: dict = None,
) -> tuple[list[dict], ClusterStats]:
    """Sammelt Tracks für einen Cluster via Seed-Artists und optionaler Radios."""
    if api_kwargs is None:
        api_kwargs = {}

    stats = ClusterStats(
        cluster_id=cluster_id,
        display_name=cluster_def["display_name"],
    )
    tracks = []
    seen_track_ids = set()

    seed_artists = cluster_def.get("seed_artists", [])
    radio_ids = cluster_def.get("radios", [])
    print(
        f"\n  Cluster: {cluster_def['display_name']} "
        f"({len(seed_artists)} Seed-Artists, {len(radio_ids)} Radios)"
    )

    # --- Seed-Artist-Scouting ---
    for artist_name in seed_artists:
        artist = search_artist(artist_name, **api_kwargs)
        if not artist:
            print(f"    Artist nicht gefunden: {artist_name}")
            if logger:
                logger.warning(f"Artist nicht gefunden: {artist_name} (Cluster: {cluster_id})")
            stats.seed_artists_not_found.append(artist_name)
            continue

        stats.seed_artists_found += 1

        top_tracks = get_artist_top_tracks(artist["id"], limit=tracks_per_artist, **api_kwargs)
        for t in top_tracks:
            track_dict = extract_track_dict(t, cluster_id)
            if track_dict and track_dict["track_id"] not in seen_track_ids:
                tracks.append(track_dict)
                seen_track_ids.add(track_dict["track_id"])

    seed_track_count = len(tracks)
    print(f"    Seeds: {stats.seed_artists_found} Artists -> {seed_track_count} Tracks")

    # --- Radio-Scouting ---
    for radio_id in radio_ids:
        before = len(seen_track_ids)
        radio_tracks_raw = get_radio_tracks(radio_id, **api_kwargs)
        added = 0
        for t in radio_tracks_raw:
            track_dict = extract_track_dict(t, cluster_id)
            if track_dict and track_dict["track_id"] not in seen_track_ids:
                tracks.append(track_dict)
                seen_track_ids.add(track_dict["track_id"])
                added += 1
        stats.radios_scouted += 1
        stats.radio_tracks_added += added
        print(f"    Radio {radio_id}: {len(radio_tracks_raw)} Tracks -> {added} neu")
        if logger:
            logger.info(
                f"Cluster {cluster_id} Radio {radio_id}: "
                f"{len(radio_tracks_raw)} Tracks, {added} neu hinzugefuegt"
            )

    if logger:
        logger.info(
            f"Cluster {cluster_id}: {stats.seed_artists_found} Artists, "
            f"{seed_track_count} Seed-Tracks, {stats.radio_tracks_added} Radio-Tracks, "
            f"{len(tracks)} gesamt, {len(stats.seed_artists_not_found)} Artists nicht gefunden"
        )

    flop_threshold = rank_thresholds.get("flop_threshold", 300000)
    hit_threshold = rank_thresholds.get("hit_threshold", 700000)

    stats.total_tracks = len(tracks)
    stats.unique_tracks = len(seen_track_ids)
    stats.hits = sum(1 for t in tracks if t["deezer_rank"] > hit_threshold)
    stats.mids = sum(1 for t in tracks if flop_threshold <= t["deezer_rank"] <= hit_threshold)
    stats.flops = sum(1 for t in tracks if t["deezer_rank"] < flop_threshold)
    stats.avg_rank = sum(t["deezer_rank"] for t in tracks) / max(1, len(tracks))

    return tracks, stats


def collect_chart_tracks(
    country_code: str,
    charts_config: dict,
    api_kwargs: dict = None,
) -> tuple[list[dict], int]:
    """Sammelt Tracks aus einer Länder-Topliste."""
    if api_kwargs is None:
        api_kwargs = {}

    if country_code not in charts_config:
        print(f"  Unbekannter Country-Code: {country_code}")
        return [], 0

    chart = charts_config[country_code]
    cluster_name = f"charts_{country_code.lower()}"
    print(f"\n  Chart: {chart['name']} (Playlist {chart['playlist_id']})")

    playlist_tracks = get_playlist_tracks(chart["playlist_id"], **api_kwargs)
    tracks = []

    for t in playlist_tracks:
        track_dict = extract_track_dict(t, cluster_name)
        if track_dict:
            tracks.append(track_dict)

    print(f"    {len(tracks)} Tracks")

    if logger:
        logger.info(f"Chart {country_code}: {len(tracks)} Tracks")

    return tracks, len(tracks)


# ══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def print_cluster_report(stats: ClusterStats, rank_thresholds: dict):
    """Gibt Cluster-Statistiken formatiert aus."""
    flop_t = rank_thresholds.get("flop_threshold", 300000)
    hit_t = rank_thresholds.get("hit_threshold", 700000)
    viability = "OK" if stats.unique_tracks >= 200 else "WARN" if stats.unique_tracks >= 100 else "LOW"
    missing = (
        ', '.join(stats.seed_artists_not_found[:3])
        + ('...' if len(stats.seed_artists_not_found) > 3 else '')
        if stats.seed_artists_not_found else '-'
    )
    radio_info = (
        f"{stats.radios_scouted} Radios (+{stats.radio_tracks_added} Tracks)"
        if stats.radios_scouted > 0 else "-"
    )

    print(f"""
    {stats.display_name}
    {'─' * 57}
    Tracks gesamt:     {stats.unique_tracks:>6}  [{viability}]
    Avg Rank:          {stats.avg_rank:>10,.0f}
    Hit (>{hit_t/1000:.0f}k):      {stats.hits:>6}  ({100*stats.hits/max(1,stats.unique_tracks):.1f}%)
    Mid ({flop_t/1000:.0f}k-{hit_t/1000:.0f}k):   {stats.mids:>6}  ({100*stats.mids/max(1,stats.unique_tracks):.1f}%)
    Flop (<{flop_t/1000:.0f}k):     {stats.flops:>6}  ({100*stats.flops/max(1,stats.unique_tracks):.1f}%)
    Seeds gefunden:    {stats.seed_artists_found:>6}
    Seeds fehlen:      {len(stats.seed_artists_not_found):>6}  {missing}
    Radio-Scouting:    {radio_info}""")


def print_summary(all_stats: list[ClusterStats], chart_tracks: int = 0):
    """Gibt Gesamtzusammenfassung aus."""
    total_tracks = sum(s.unique_tracks for s in all_stats) + chart_tracks
    viable = [s for s in all_stats if s.unique_tracks >= 200]
    marginal = [s for s in all_stats if 100 <= s.unique_tracks < 200]
    insufficient = [s for s in all_stats if s.unique_tracks < 100]

    print(f"""
{'=' * 79}
  DEEZER SCOUTING SUMMARY
{'=' * 79}
  Cluster gesamt:           {len(all_stats):>5}
  Tracks gesamt:            {total_tracks:>5}  (inkl. {chart_tracks} Chart-Tracks)

  Trainierbar (>=200):      {len(viable):>5}  {', '.join(s.cluster_id for s in viable) if viable else '-'}
  Grenzwertig (100-199):    {len(marginal):>5}  {', '.join(s.cluster_id for s in marginal) if marginal else '-'}
  Unzureichend (<100):      {len(insufficient):>5}  {', '.join(s.cluster_id for s in insufficient) if insufficient else '-'}
{'=' * 79}
""")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global logger

    # Config laden
    paths = load_paths_config()
    thresholds_cfg = load_thresholds_config()
    deezer_thresholds = thresholds_cfg.get("deezer", {})
    clusters_cfg = load_clusters_config()

    genre_clusters = get_genre_clusters(clusters_cfg)
    charts_config = get_charts_config(clusters_cfg)
    scouting_cfg = get_scouting_config(clusters_cfg)

    tracks_per_artist = scouting_cfg.get("tracks_per_artist", 25)
    request_delay = scouting_cfg.get("request_delay", 0.25)
    max_retries = scouting_cfg.get("max_retries", 3)

    api_kwargs = {"request_delay": request_delay, "max_retries": max_retries}

    parser = argparse.ArgumentParser(
        description="Scouting-Script für Spotilyzer Genre-Cluster (Deezer API)"
    )
    parser.add_argument(
        "--clusters",
        nargs="+",
        choices=list(genre_clusters.keys()) + ["all"],
        default=["all"],
        help="Cluster zum Scouten (default: all)"
    )
    parser.add_argument(
        "--charts",
        nargs="+",
        choices=list(charts_config.keys()) + ["none"],
        default=["none"],
        help="Laender-Charts einbeziehen (default: none)"
    )
    parser.add_argument(
        "--tracks-per-artist",
        type=int,
        default=tracks_per_artist,
        help=f"Tracks pro Artist (default: {tracks_per_artist}, max: 50)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Zeige was gescoutet wuerde, ohne API-Aufrufe"
    )
    args = parser.parse_args()

    # Logging
    logger = setup_logging("scout", log_dir=paths.get("logs"))

    tracks_per_artist = min(50, args.tracks_per_artist)

    if "all" in args.clusters:
        selected_clusters = list(genre_clusters.keys())
    else:
        selected_clusters = args.clusters

    # Metadata-Verzeichnis sicherstellen
    metadata_dir = paths.get("metadata")
    ensure_dir(metadata_dir)
    jsonl_path = get_tracks_jsonl_path(metadata_dir)

    all_stats = []
    all_tracks = []
    chart_track_count = 0

    print(f"{'=' * 79}")
    print(f"  SPOTILYZER GENRE SCOUTING (Deezer API)")
    print(f"  Cluster: {len(selected_clusters)} | Tracks/Artist: {tracks_per_artist}")
    print(f"  Charts: {', '.join(args.charts) if 'none' not in args.charts else 'keine'}")
    print(f"  Output: {jsonl_path}")
    print(f"{'=' * 79}")

    logger.info(
        f"Scouting gestartet: {len(selected_clusters)} Cluster, "
        f"{tracks_per_artist} Tracks/Artist"
    )

    if args.dry_run:
        print(f"\n  [DRY RUN] Keine API-Aufrufe.")
        for cid in selected_clusters:
            cdef = genre_clusters[cid]
            n_seeds = len(cdef.get("seed_artists", []))
            n_radios = len(cdef.get("radios", []))
            radio_str = f", {n_radios} Radios" if n_radios else ""
            print(f"    {cid}: {n_seeds} Seed-Artists{radio_str}")
        return

    # Genre-Cluster scouten
    for cluster_id in selected_clusters:
        cluster_def = genre_clusters[cluster_id]
        tracks, stats = collect_tracks_for_cluster(
            cluster_id, cluster_def, tracks_per_artist,
            deezer_thresholds, api_kwargs,
        )

        all_stats.append(stats)
        all_tracks.extend(tracks)

        print_cluster_report(stats, deezer_thresholds)

    # Länder-Charts scouten (optional)
    if "none" not in args.charts:
        print(f"\n{'─' * 79}")
        print(f"  LAENDER-CHARTS")
        print(f"{'─' * 79}")

        for country in args.charts:
            chart_tracks, count = collect_chart_tracks(country, charts_config, api_kwargs)
            all_tracks.extend(chart_tracks)
            chart_track_count += count

    # Summary
    print_summary(all_stats, chart_track_count)

    # In JSONL mergen
    if all_tracks:
        updated, added = merge_tracks(jsonl_path, all_tracks)
        print(f"  JSONL aktualisiert: {jsonl_path}")
        print(f"    Neu:          {added}")
        print(f"    Aktualisiert: {updated}")
        print(f"    Gesamt:       {len(read_tracks(jsonl_path))}")

        logger.info(f"JSONL gespeichert: {added} neu, {updated} aktualisiert -> {jsonl_path}")

    logger.info(
        f"Scouting abgeschlossen: {len(all_tracks)} Tracks, "
        f"{chart_track_count} Chart-Tracks"
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
