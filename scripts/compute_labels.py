"""
compute_labels.py
==================
Berechnet Multi-Source-Labels aus Deezer-Rank + Last.fm-Metriken.

Workflow:
1. Liest scouted_tracks_enriched.csv
2. Berechnet Einzelsignale (Deezer, Last.fm)
3. Bestimmt Konsens-Label und Robustheit
4. Berechnet Sample-Gewichte für Training
5. Speichert labeled_tracks.csv

Autor: Claude (für Andreas Vogelsang / Spotilyzer)
Datum: 2026-03-07
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import yaml


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
    
    # Beide Bedingungen für Hit
    if playcount > hit_playcount and listeners > hit_listeners:
        return "hit"
    
    # Eine Bedingung für Flop reicht
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
    
    # Dissens → konservativ zur Mitte
    return "mid", "contested"


def compute_sample_weight(robustness: str, weights: dict) -> float:
    """Berechnet Sample-Gewicht basierend auf Robustheit."""
    return weights.get(robustness, 0.5)


def compute_composite_score(
    hit_probability: float,
    robustness: str,
    robustness_factors: dict,
) -> float:
    """Berechnet Composite Score für UI-Sortierung."""
    factor = robustness_factors.get(robustness, 0.7)
    return hit_probability * factor


# ══════════════════════════════════════════════════════════════════════════════
# HAUPTLOGIK
# ══════════════════════════════════════════════════════════════════════════════

def process_labels(
    input_path: Path,
    output_path: Path,
    config: dict,
) -> dict:
    """
    Verarbeitet Tracks und berechnet Labels.
    
    Returns:
        Statistiken
    """
    # Config extrahieren
    deezer_cfg = config["deezer"]
    lastfm_cfg = config["lastfm"]
    weights_cfg = config["sample_weights"]
    
    # Daten laden
    df = pd.read_csv(input_path)
    print(f"Geladen: {len(df)} Tracks aus {input_path}")
    
    # Neue Spalten
    labels = []
    robustness_values = []
    sample_weights = []
    deezer_signals = []
    lastfm_signals = []
    
    # Statistiken
    stats = {
        "total": len(df),
        "validated": 0,
        "single_source": 0,
        "contested": 0,
        "hits": 0,
        "mids": 0,
        "flops": 0,
    }
    
    for idx, row in df.iterrows():
        # Deezer-Signal
        deezer_signal = rank_to_signal(
            deezer_rank=row["rank"],
            hit_threshold=deezer_cfg["hit_threshold"],
            flop_threshold=deezer_cfg["flop_threshold"],
        )
        deezer_signals.append(deezer_signal)
        
        # Last.fm-Signal (falls verfügbar)
        if row.get("lastfm_matched", False):
            lastfm_signal = plays_to_signal(
                playcount=row.get("lastfm_playcount"),
                listeners=row.get("lastfm_listeners"),
                hit_playcount=lastfm_cfg["hit_playcount"],
                hit_listeners=lastfm_cfg["hit_listeners"],
                flop_playcount=lastfm_cfg["flop_playcount"],
                flop_listeners=lastfm_cfg["flop_listeners"],
            )
        else:
            lastfm_signal = None
        lastfm_signals.append(lastfm_signal)
        
        # Label und Robustheit
        label, robustness = compute_label_and_robustness(deezer_signal, lastfm_signal)
        labels.append(label)
        robustness_values.append(robustness)
        
        # Sample Weight
        weight = compute_sample_weight(robustness, weights_cfg)
        sample_weights.append(weight)
        
        # Statistiken
        stats[robustness] += 1
        stats[f"{label}s"] += 1
    
    # Spalten hinzufügen
    df["deezer_signal"] = deezer_signals
    df["lastfm_signal"] = lastfm_signals
    df["label"] = labels
    df["robustness"] = robustness_values
    df["sample_weight"] = sample_weights
    
    # Speichern
    df.to_csv(output_path, index=False)
    print(f"Gespeichert: {output_path}")
    
    return stats


def print_stats(stats: dict):
    """Gibt Statistiken formatiert aus."""
    print(f"""
╔═══════════════════════════════════════════════════════════════════════════════
║  LABEL COMPUTATION COMPLETE
╠═══════════════════════════════════════════════════════════════════════════════
║  Tracks gesamt:        {stats['total']:>6}
╠═══════════════════════════════════════════════════════════════════════════════
║  ROBUSTHEIT:
║  ✅ Validated:         {stats['validated']:>6}  ({100*stats['validated']/stats['total']:.1f}%)
║  ⚠️  Single-Source:     {stats['single_source']:>6}  ({100*stats['single_source']/stats['total']:.1f}%)
║  ❓ Contested:         {stats['contested']:>6}  ({100*stats['contested']/stats['total']:.1f}%)
╠═══════════════════════════════════════════════════════════════════════════════
║  LABELS:
║  🏆 Hits:              {stats['hits']:>6}  ({100*stats['hits']/stats['total']:.1f}%)
║  📊 Mids:              {stats['mids']:>6}  ({100*stats['mids']/stats['total']:.1f}%)
║  📉 Flops:             {stats['flops']:>6}  ({100*stats['flops']/stats['total']:.1f}%)
╚═══════════════════════════════════════════════════════════════════════════════
""")


def print_cross_tabulation(input_path: Path):
    """Zeigt Kreuztabelle Deezer vs. Last.fm Signale."""
    df = pd.read_csv(input_path)
    
    # Nur Tracks mit Last.fm-Match
    matched = df[df["lastfm_signal"].notna()]
    
    if len(matched) == 0:
        print("Keine Tracks mit Last.fm-Match für Kreuztabelle.")
        return
    
    cross = pd.crosstab(
        matched["deezer_signal"],
        matched["lastfm_signal"],
        margins=True,
    )
    
    print("\n📊 Kreuztabelle: Deezer vs. Last.fm Signale")
    print("=" * 50)
    print(cross)
    print("")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Berechnet Multi-Source-Labels für Training"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("data/scouted_tracks_enriched.csv"),
        help="Input CSV (mit Last.fm-Daten)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/labeled_tracks.csv"),
        help="Output CSV",
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path("configs/thresholds.yaml"),
        help="Config-Datei mit Schwellenwerten",
    )
    parser.add_argument(
        "--show-crosstab",
        action="store_true",
        help="Zeige Kreuztabelle Deezer vs. Last.fm",
    )
    args = parser.parse_args()
    
    # Config laden
    if not args.config.exists():
        print(f"Fehler: Config-Datei nicht gefunden: {args.config}")
        return
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Input prüfen
    if not args.input.exists():
        print(f"Fehler: Input-Datei nicht gefunden: {args.input}")
        print("Bitte zuerst enrich_lastfm.py ausführen.")
        return
    
    # Verarbeitung
    stats = process_labels(
        input_path=args.input,
        output_path=args.output,
        config=config,
    )
    
    print_stats(stats)
    
    if args.show_crosstab:
        print_cross_tabulation(args.output)


if __name__ == "__main__":
    main()
