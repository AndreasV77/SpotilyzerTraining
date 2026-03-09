"""
playlist.py
===========
M3U8-Playlist-Generierung aus JSONL-Metadaten.

Erzeugt Extended M3U Playlists für lesbare Tracklisten.
"""

from pathlib import Path


def create_playlist(
    tracks: list[dict],
    output_path: Path,
    base_path: str = "",
):
    """
    Erstellt eine M3U8-Playlist aus Track-Dicts.

    Args:
        tracks: Liste von Track-Dicts (braucht file_path, title, artist)
        output_path: Pfad für die .m3u8-Datei
        base_path: Optionaler Prefix für Dateipfade
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("#EXTM3U\n")

        for track in tracks:
            file_path = track.get("file_path", "")
            if not file_path:
                continue

            artist = track.get("artist", "Unknown")
            title = track.get("title", "Unknown")

            if base_path:
                full_path = f"{base_path}/{file_path}"
            else:
                full_path = file_path

            f.write(f"#EXTINF:30,{artist} - {title}\n")
            f.write(f"{full_path}\n")


def find_track(tracks: list[dict], query: str) -> list[dict]:
    """
    Sucht Tracks nach Artist oder Titel (case-insensitive).

    Args:
        tracks: Liste von Track-Dicts
        query: Suchbegriff

    Returns:
        Gefilterte Liste
    """
    query_lower = query.lower()
    return [
        t for t in tracks
        if query_lower in t.get("title", "").lower()
        or query_lower in t.get("artist", "").lower()
    ]
