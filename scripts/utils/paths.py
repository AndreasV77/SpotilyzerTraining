"""
paths.py
========
Pfad-Hilfsfunktionen für MD5-Sharding und Preview-Dateien.
"""

import hashlib
from pathlib import Path


def get_shard_dir(track_id: int) -> str:
    """
    Berechnet Shard-Verzeichnis aus Track-ID.

    Verwendet die ersten 2 Zeichen des MD5-Hashs der Track-ID.
    Ergibt 256 mögliche Unterverzeichnisse (00-ff).

    Args:
        track_id: Deezer Track-ID

    Returns:
        Zweistelliger Hex-String (z.B. "a7", "e1")
    """
    h = hashlib.md5(str(track_id).encode()).hexdigest()
    return h[:2]


def get_preview_path(track_id: int, base_path: str | Path) -> Path:
    """
    Vollständiger Pfad zu einer Preview-Datei.

    Args:
        track_id: Deezer Track-ID
        base_path: Basis-Verzeichnis für Previews (z.B. G:/Dev/SpotilyzerData/previews)

    Returns:
        Path-Objekt zum MP3 (z.B. previews/a7/3770028292.mp3)
    """
    shard = get_shard_dir(track_id)
    return Path(base_path) / shard / f"{track_id}.mp3"


def get_relative_preview_path(track_id: int) -> str:
    """
    Relativer Pfad zur Preview-Datei (für JSONL file_path-Feld).

    Returns:
        String wie "previews/a7/3770028292.mp3"
    """
    shard = get_shard_dir(track_id)
    return f"previews/{shard}/{track_id}.mp3"


def ensure_shard_dir(track_id: int, base_path: str | Path) -> Path:
    """
    Erstellt das Shard-Verzeichnis falls nötig und gibt es zurück.

    Args:
        track_id: Deezer Track-ID
        base_path: Basis-Verzeichnis für Previews

    Returns:
        Path zum Shard-Verzeichnis
    """
    shard = get_shard_dir(track_id)
    shard_dir = Path(base_path) / shard
    shard_dir.mkdir(parents=True, exist_ok=True)
    return shard_dir
