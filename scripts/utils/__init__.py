"""
utils — Gemeinsame Hilfsfunktionen für die SpotilyzerTraining-Pipeline.

Module:
  paths.py     — MD5-Sharding, Pfad-Resolution
  metadata.py  — JSONL lesen/schreiben/updaten
  playlist.py  — M3U8-Playlist-Generierung
"""

from .paths import get_shard_dir, get_preview_path
from .metadata import read_tracks, write_tracks, update_tracks, append_tracks
