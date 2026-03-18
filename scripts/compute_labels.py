"""
compute_labels.py
==================
Berechnet Multi-Source-Labels aus Deezer-Rank + Last.fm-Metriken.

Workflow:
1. Liest metadata/tracks.jsonl
2. Berechnet Einzelsignale (Deezer, Last.fm)
3. Bestimmt Konsens-Label und Robustheit
4. Schreibt label + robustness zurueck in JSONL
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple

from _utils import (
    setup_logging,
    load_paths_config,
    load_thresholds_config,
)
from utils.metadata import (
    read_tracks,
    update_tracks,
    get_tracks_jsonl_path,
)

# Logger (wird in main() initialisiert)
logger = None


# ══════════════════════════════════════════════════════════════════════════════
# LABEL-LOGIK
# ══════════════════════════════════════════════════════════════════════════════

def rank_to_signal(
    deezer_rank: int,
    hit_threshold: int,
    flop_threshold: int,
) -> str:
    """Konvertiert Deezer-Rank zu Signal."""
    if deezer_rank > hit_threshold:
        return "hit"
    elif deezer_rank < flop_threshold:
        return "flop"
    return "mid"


def plays_to_signal(
    playcount: Optional[int],
    listeners: Optional[int],
    hit_playcount: int,
    hit_listeners: int,
    flop_playcount: int,
    flop_listeners: int,
) -> Optional[str]:
    """Konvertiert Last.fm-Metriken zu Signal."""
    if playcount is None or listeners is None:
        return None

    # Beide Bedingungen fuer Hit
    if playcount > hit_playcount and listeners > hit_listeners:
        return "hit"

    # Eine Bedingung fuer Flop reicht
    if playcount < flop_playcount or listeners < flop_listeners:
        return "flop"

    return "mid"


def compute_label_and_robustness(
    deezer_signal: str,
    lastfm_signal: Optional[str],
) -> Tuple[str, str]:
    """
    Bestimmt finales Label und Robustheit.

    Returns:
        (label, robustness)
        - label: "hit" / "mid" / "flop"
        - robustness: "validated" / "single_source" / "contested"
    """
    if lastfm_signal is None:
        return deezer_signal, "single_source"

    if deezer_signal == lastfm_signal:
        return deezer_signal, "validated"

    # Extremer Widerspruch (hit vs. flop) → konservativ zur Mitte
    if {deezer_signal, lastfm_signal} == {"hit", "flop"}:
        return "mid", "contested"

    # Leichter Widerspruch (hit vs. mid oder flop vs. mid) → Deezer bevorzugen
    # Deezer-Rank ist track-spezifisch und aktueller als Last.fm-Histogramdaten
    return deezer_signal, "contested"


# ══════════════════════════════════════════════════════════════════════════════
# HAUPTLOGIK
# ══════════════════════════════════════════════════════════════════════════════

def process_labels(
    jsonl_path: Path,
    config: dict,
    resume: bool = True,
) -> dict:
    """
    Verarbeitet Tracks und berechnet Labels.

    Returns:
        Statistiken
    """
    # Config extrahieren
    deezer_cfg = config["deezer"]
    lastfm_cfg = config["lastfm"]

    # Tracks laden
    tracks = read_tracks(jsonl_path)
    print(f"  Geladen: {len(tracks)} Tracks aus {jsonl_path}")

    # Statistiken
    stats = {
        "total": len(tracks),
        "processed": 0,
        "skipped": 0,
        "validated": 0,
        "single_source": 0,
        "contested": 0,
        "hits": 0,
        "mids": 0,
        "flops": 0,
        "no_rank": 0,
    }

    updates = {}

    for track in tracks:
        track_id = track["track_id"]

        # Resume: Skip wenn bereits gelabelt
        if resume and track.get("label") is not None:
            stats["skipped"] += 1
            continue

        deezer_rank = track.get("deezer_rank")
        if deezer_rank is None:
            stats["no_rank"] += 1
            if logger:
                logger.warning(f"Track {track_id}: kein deezer_rank")
            continue

        # Deezer-Signal
        deezer_signal = rank_to_signal(
            deezer_rank=deezer_rank,
            hit_threshold=deezer_cfg["hit_threshold"],
            flop_threshold=deezer_cfg["flop_threshold"],
        )

        # Last.fm-Signal (falls verfuegbar)
        lastfm_playcount = track.get("lastfm_playcount")
        lastfm_listeners = track.get("lastfm_listeners")

        if lastfm_playcount is not None and lastfm_listeners is not None:
            lastfm_signal = plays_to_signal(
                playcount=lastfm_playcount,
                listeners=lastfm_listeners,
                hit_playcount=lastfm_cfg["hit_playcount"],
                hit_listeners=lastfm_cfg["hit_listeners"],
                flop_playcount=lastfm_cfg["flop_playcount"],
                flop_listeners=lastfm_cfg["flop_listeners"],
            )
        else:
            lastfm_signal = None

        # Label und Robustheit
        label, robustness = compute_label_and_robustness(deezer_signal, lastfm_signal)

        stats["processed"] += 1
        stats[robustness] += 1
        stats[f"{label}s"] += 1

        updates[track_id] = {
            "label": label,
            "robustness": robustness,
        }

    # In JSONL zurueckschreiben
    if updates:
        update_tracks(jsonl_path, updates)

    return stats


def print_stats(stats: dict):
    """Gibt Statistiken formatiert aus."""
    total = max(1, stats["total"])
    processed = max(1, stats["processed"])
    print(f"""
{'=' * 79}
  LABEL COMPUTATION COMPLETE
{'=' * 79}
  Tracks gesamt:        {stats['total']:>6}
  Bereits gelabelt:     {stats['skipped']:>6}
  Neu verarbeitet:      {stats['processed']:>6}
  Ohne Rank:            {stats['no_rank']:>6}
{'─' * 79}
  ROBUSTHEIT:
    Validated:          {stats['validated']:>6}  ({100*stats['validated']/processed:.1f}%)
    Single-Source:      {stats['single_source']:>6}  ({100*stats['single_source']/processed:.1f}%)
    Contested:          {stats['contested']:>6}  ({100*stats['contested']/processed:.1f}%)
{'─' * 79}
  LABELS:
    Hits:               {stats['hits']:>6}  ({100*stats['hits']/processed:.1f}%)
    Mids:               {stats['mids']:>6}  ({100*stats['mids']/processed:.1f}%)
    Flops:              {stats['flops']:>6}  ({100*stats['flops']/processed:.1f}%)
{'=' * 79}
""")


def print_cross_tabulation(jsonl_path: Path, config: dict):
    """Zeigt Kreuztabelle Deezer vs. Last.fm Signale."""
    deezer_cfg = config["deezer"]
    lastfm_cfg = config["lastfm"]

    tracks = read_tracks(jsonl_path)

    # Zaehler fuer Kreuztabelle
    cross = {}
    for track in tracks:
        deezer_rank = track.get("deezer_rank")
        if deezer_rank is None:
            continue

        deezer_signal = rank_to_signal(
            deezer_rank, deezer_cfg["hit_threshold"], deezer_cfg["flop_threshold"]
        )

        lastfm_playcount = track.get("lastfm_playcount")
        lastfm_listeners = track.get("lastfm_listeners")

        if lastfm_playcount is not None and lastfm_listeners is not None:
            lastfm_signal = plays_to_signal(
                lastfm_playcount, lastfm_listeners,
                lastfm_cfg["hit_playcount"], lastfm_cfg["hit_listeners"],
                lastfm_cfg["flop_playcount"], lastfm_cfg["flop_listeners"],
            )
        else:
            continue  # Nur Tracks mit Last.fm-Match

        key = (deezer_signal, lastfm_signal)
        cross[key] = cross.get(key, 0) + 1

    if not cross:
        print("  Keine Tracks mit Last.fm-Match fuer Kreuztabelle.")
        return

    signals = ["hit", "mid", "flop"]

    print(f"\n  Kreuztabelle: Deezer vs. Last.fm Signale")
    print(f"  {'':12} {'hit':>8} {'mid':>8} {'flop':>8} {'Total':>8}")
    print(f"  {'─' * 48}")

    for d_sig in signals:
        row_total = sum(cross.get((d_sig, l_sig), 0) for l_sig in signals)
        row = f"  {d_sig:12}"
        for l_sig in signals:
            count = cross.get((d_sig, l_sig), 0)
            row += f" {count:>8}"
        row += f" {row_total:>8}"
        print(row)

    # Totals
    print(f"  {'─' * 48}")
    total_row = f"  {'Total':12}"
    for l_sig in signals:
        col_total = sum(cross.get((d_sig, l_sig), 0) for d_sig in signals)
        total_row += f" {col_total:>8}"
    grand_total = sum(cross.values())
    total_row += f" {grand_total:>8}"
    print(total_row)
    print()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global logger

    paths = load_paths_config()
    metadata_dir = paths.get("metadata")
    jsonl_path = get_tracks_jsonl_path(metadata_dir)

    parser = argparse.ArgumentParser(
        description="Berechnet Multi-Source-Labels fuer Training (JSONL)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Nicht fortsetzen, alle Tracks neu labeln",
    )
    parser.add_argument(
        "--show-crosstab",
        action="store_true",
        help="Zeige Kreuztabelle Deezer vs. Last.fm",
    )
    args = parser.parse_args()

    # Logging
    logger = setup_logging("labels", log_dir=paths.get("logs"))

    # Config laden
    config = load_thresholds_config()

    # Input pruefen
    if not jsonl_path.exists():
        print(f"Fehler: tracks.jsonl nicht gefunden: {jsonl_path}")
        print("  -> Erst 'python scripts/scout_deezer.py' ausfuehren!")
        sys.exit(1)

    print(f"{'=' * 79}")
    print(f"  LABEL COMPUTATION")
    print(f"{'=' * 79}")
    print(f"  JSONL:   {jsonl_path}")
    print(f"  Resume:  {'Nein' if args.no_resume else 'Ja'}")

    logger.info(f"Label-Berechnung gestartet: {jsonl_path}")

    # Verarbeitung
    stats = process_labels(
        jsonl_path=jsonl_path,
        config=config,
        resume=not args.no_resume,
    )

    print_stats(stats)

    if args.show_crosstab:
        print_cross_tabulation(jsonl_path, config)

    logger.info(
        f"Label-Berechnung abgeschlossen: {stats['hits']} hits, "
        f"{stats['mids']} mids, {stats['flops']} flops "
        f"({stats['validated']} validated, {stats['contested']} contested)"
    )


if __name__ == "__main__":
    main()
