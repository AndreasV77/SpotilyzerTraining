"""
metadata.py
===========
JSONL-Metadaten lesen/schreiben/updaten.

tracks.jsonl ist die zentrale Datenquelle für alle Pipeline-Schritte.
Jede Zeile ist ein JSON-Objekt mit track_id als Primärschlüssel.

Format:
    {"track_id": 123, "title": "...", "artist": "...", "clusters": [...], ...}

Jeder Pipeline-Schritt fügt Felder hinzu:
    scout:     track_id, title, artist, album, clusters, deezer_rank
    download:  file_path
    enrich:    lastfm_playcount, lastfm_listeners, lastfm_tags
    labels:    label, robustness, sample_weight
"""

import json
from pathlib import Path
from typing import Optional


def read_tracks(jsonl_path: Path) -> list[dict]:
    """
    Liest alle Tracks aus einer JSONL-Datei.

    Args:
        jsonl_path: Pfad zur tracks.jsonl

    Returns:
        Liste von Track-Dicts
    """
    tracks = []
    if not jsonl_path.exists():
        return tracks

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                tracks.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNUNG: Zeile {line_num} ungültig: {e}")

    return tracks


def read_tracks_as_dict(jsonl_path: Path) -> dict[int, dict]:
    """
    Liest Tracks als Dict mit track_id als Key.

    Returns:
        {track_id: track_dict, ...}
    """
    tracks = read_tracks(jsonl_path)
    return {t["track_id"]: t for t in tracks if "track_id" in t}


def write_tracks(jsonl_path: Path, tracks: list[dict]):
    """
    Schreibt alle Tracks in eine JSONL-Datei (überschreibt!).

    Args:
        jsonl_path: Pfad zur tracks.jsonl
        tracks: Liste von Track-Dicts
    """
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for track in tracks:
            f.write(json.dumps(track, ensure_ascii=False) + "\n")


def append_tracks(jsonl_path: Path, new_tracks: list[dict]):
    """
    Fügt neue Tracks ans Ende der JSONL-Datei an.
    Prüft NICHT auf Duplikate — dafür merge_tracks verwenden.

    Args:
        jsonl_path: Pfad zur tracks.jsonl
        new_tracks: Neue Track-Dicts
    """
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with open(jsonl_path, "a", encoding="utf-8") as f:
        for track in new_tracks:
            f.write(json.dumps(track, ensure_ascii=False) + "\n")


def update_tracks(
    jsonl_path: Path,
    updates: dict[int, dict],
):
    """
    Aktualisiert bestehende Tracks mit neuen Feldern.

    Liest die JSONL, merged Updates rein, schreibt zurück.
    Tracks ohne Update bleiben unverändert.

    Args:
        jsonl_path: Pfad zur tracks.jsonl
        updates: {track_id: {feld: wert, ...}, ...}
    """
    tracks = read_tracks(jsonl_path)

    for track in tracks:
        tid = track.get("track_id")
        if tid in updates:
            track.update(updates[tid])

    write_tracks(jsonl_path, tracks)


def merge_tracks(
    jsonl_path: Path,
    new_tracks: list[dict],
) -> tuple[int, int]:
    """
    Merged neue Tracks in bestehende JSONL.

    - Bestehende Tracks: clusters werden gemerged (Union), andere Felder aktualisiert
    - Neue Tracks: werden angehängt

    Args:
        jsonl_path: Pfad zur tracks.jsonl
        new_tracks: Neue Track-Dicts

    Returns:
        (updated_count, added_count)
    """
    existing = read_tracks_as_dict(jsonl_path)
    updated = 0
    added = 0

    for new_track in new_tracks:
        tid = new_track.get("track_id")
        if tid is None:
            continue

        if tid in existing:
            # Clusters mergen (Union)
            old_clusters = set(existing[tid].get("clusters", []))
            new_clusters = set(new_track.get("clusters", []))
            merged_clusters = sorted(old_clusters | new_clusters)

            existing[tid].update(new_track)
            existing[tid]["clusters"] = merged_clusters
            updated += 1
        else:
            existing[tid] = new_track
            added += 1

    # Zurückschreiben (Reihenfolge nach track_id)
    all_tracks = [existing[tid] for tid in sorted(existing.keys())]
    write_tracks(jsonl_path, all_tracks)

    return updated, added


def get_tracks_jsonl_path(metadata_dir: Path) -> Path:
    """Gibt den Standard-Pfad zur tracks.jsonl zurück."""
    return metadata_dir / "tracks.jsonl"


def filter_tracks(
    tracks: list[dict],
    has_field: Optional[str] = None,
    missing_field: Optional[str] = None,
    label: Optional[str] = None,
    cluster: Optional[str] = None,
    has_file: bool = False,
) -> list[dict]:
    """
    Filtert eine Track-Liste nach verschiedenen Kriterien.

    Args:
        tracks: Liste von Track-Dicts
        has_field: Nur Tracks mit diesem Feld (nicht None/leer)
        missing_field: Nur Tracks OHNE dieses Feld
        label: Nur Tracks mit diesem Label
        cluster: Nur Tracks in diesem Cluster
        has_file: Nur Tracks mit gesetztem file_path

    Returns:
        Gefilterte Liste
    """
    result = tracks

    if has_field:
        result = [t for t in result if t.get(has_field) is not None]

    if missing_field:
        result = [t for t in result if t.get(missing_field) is None]

    if label:
        result = [t for t in result if t.get("label") == label]

    if cluster:
        result = [t for t in result if cluster in t.get("clusters", [])]

    if has_file:
        result = [t for t in result if t.get("file_path")]

    return result
