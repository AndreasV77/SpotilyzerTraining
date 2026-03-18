"""
inspect_dataset.py
==================
Lese-only Diagnose-Tool für tracks.jsonl.

Bewertet die Qualität und Balance der Trainingsdaten BEVOR Änderungen
(Balancierung, Cluster-Erweiterung) vorgenommen werden.

Kein Schreiben außer optionalem Report, keine API-Calls.

Sektionen:
  1. Pipeline-Status     — Wie weit sind die Daten aufbereitet?
  2. Label-Verteilung    — Hit/Mid/Flop global
  3. Robustheit          — Validated/Single-source/Contested je Label
  4. Training-Qualität   — Was würde das Modell tatsächlich sehen?
  5. Per-Cluster         — Verteilung je Genre-Cluster, sortiert nach Nützlichkeit
  6. Overlap             — Tracks in mehreren Clustern
  7. Embedding-Status    — Schnittmenge Labels ∩ Embeddings (falls vorhanden)

Verwendung:
  python scripts/inspect_dataset.py
  python scripts/inspect_dataset.py --report
  python scripts/inspect_dataset.py --cluster extreme_metal
  python scripts/inspect_dataset.py --min-flop-pct 10
  python scripts/inspect_dataset.py --validated-only
  python scripts/inspect_dataset.py --skip-embeddings
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from _utils import load_paths_config, ensure_dir
from utils.metadata import read_tracks, get_tracks_jsonl_path


# ══════════════════════════════════════════════════════════════════════════════
# KONSTANTEN
# ══════════════════════════════════════════════════════════════════════════════

LABELS = ["hit", "mid", "flop"]
ROBUSTNESS_LEVELS = ["validated", "single_source", "contested"]

# Schwellenwerte für Warnungen (überschreibbar per --min-flop-pct)
DEFAULT_MIN_FLOP_PCT = 10.0       # Unter 10% Flop → Warnung
DEFAULT_MIN_LABELED_PCT = 80.0    # Unter 80% gelabelt → Hinweis
DEFAULT_MIN_VALIDATED_PCT = 30.0  # Unter 30% validated → Hinweis


# ══════════════════════════════════════════════════════════════════════════════
# HILFSFUNKTIONEN
# ══════════════════════════════════════════════════════════════════════════════

def pct(n: int, total: int) -> str:
    """Formatiert n/total als Prozent-String."""
    if total == 0:
        return "  -  "
    return f"{100.0 * n / total:5.1f}%"


def bar(n: int, total: int, width: int = 20) -> str:
    """Einfacher ASCII-Balken."""
    if total == 0:
        return " " * width
    filled = round(width * n / total)
    return "█" * filled + "░" * (width - filled)


def imbalance_ratio(counts: dict) -> str:
    """
    Gibt das Imbalance-Verhältnis als lesbaren String zurück.
    Beispiel: "Mid 4.2× häufiger als Flop"
    """
    labeled = {k: counts.get(k, 0) for k in LABELS}
    if labeled["flop"] == 0:
        return "WARNUNG: Keine Flop-Samples!"
    max_label = max(labeled, key=labeled.get)
    ratio = labeled[max_label] / labeled["flop"]
    if ratio <= 1.5:
        return f"Gut ausgewogen (max. {ratio:.1f}× Spread)"
    return f"{max_label.capitalize()} ist {ratio:.1f}× häufiger als Flop"


def warn(msg: str):
    print(f"  ⚠  {msg}")


def info(msg: str):
    print(f"  →  {msg}")


# ══════════════════════════════════════════════════════════════════════════════
# SEKTION 1: PIPELINE-STATUS
# ══════════════════════════════════════════════════════════════════════════════

def section_pipeline_status(tracks: list[dict], min_labeled_pct: float) -> dict:
    print(f"\n{'─' * 79}")
    print(f"  1. PIPELINE-STATUS")
    print(f"{'─' * 79}")

    total = len(tracks)
    has_file      = sum(1 for t in tracks if t.get("file_path"))
    has_lastfm    = sum(1 for t in tracks if t.get("lastfm_playcount") is not None)
    has_label     = sum(1 for t in tracks if t.get("label") is not None)
    has_file_and_label = sum(
        1 for t in tracks if t.get("file_path") and t.get("label") is not None
    )

    print(f"  Tracks gesamt:              {total:>6}")
    print(f"  Mit Audio (file_path):      {has_file:>6}  ({pct(has_file, total)})")
    print(f"  Mit Last.fm-Daten:          {has_lastfm:>6}  ({pct(has_lastfm, total)})")
    print(f"  Mit Label:                  {has_label:>6}  ({pct(has_label, total)})")
    print(f"  Mit Audio + Label:          {has_file_and_label:>6}  ({pct(has_file_and_label, total)})")

    labeled_pct = 100.0 * has_label / total if total > 0 else 0
    if labeled_pct < min_labeled_pct:
        warn(f"Nur {labeled_pct:.0f}% der Tracks sind gelabelt — Labels berechnen!")

    return {
        "total": total,
        "has_file": has_file,
        "has_lastfm": has_lastfm,
        "has_label": has_label,
        "has_file_and_label": has_file_and_label,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SEKTION 2: LABEL-VERTEILUNG
# ══════════════════════════════════════════════════════════════════════════════

def section_label_distribution(labeled_tracks: list[dict], min_flop_pct: float) -> Counter:
    print(f"\n{'─' * 79}")
    print(f"  2. LABEL-VERTEILUNG  (n={len(labeled_tracks)})")
    print(f"{'─' * 79}")

    counts = Counter(t["label"] for t in labeled_tracks)
    total = len(labeled_tracks)

    print(f"  {'Label':8} {'Anzahl':>7}  {'Anteil':>7}  Balken")
    print(f"  {'─' * 55}")
    for label in LABELS:
        n = counts.get(label, 0)
        print(f"  {label:8} {n:>7}  {pct(n, total)}  {bar(n, total)}")

    print()
    print(f"  Imbalance: {imbalance_ratio(counts)}")

    flop_pct = 100.0 * counts.get("flop", 0) / total if total > 0 else 0
    if flop_pct < min_flop_pct:
        warn(f"Flop-Anteil {flop_pct:.1f}% < {min_flop_pct:.0f}% — Imbalance-Problem!")

    return counts  # Counter ist direkt JSON-serialisierbar (via dict())


# ══════════════════════════════════════════════════════════════════════════════
# SEKTION 3: ROBUSTHEIT
# ══════════════════════════════════════════════════════════════════════════════

def section_robustness(labeled_tracks: list[dict], min_validated_pct: float) -> dict:
    print(f"\n{'─' * 79}")
    print(f"  3. ROBUSTHEIT")
    print(f"{'─' * 79}")

    total = len(labeled_tracks)
    rob_counts = Counter(t.get("robustness", "single_source") for t in labeled_tracks)

    print(f"  Gesamt:")
    print(f"  {'Robustheit':18} {'Anzahl':>7}  {'Anteil':>7}  Balken")
    print(f"  {'─' * 55}")
    for rob in ROBUSTNESS_LEVELS:
        n = rob_counts.get(rob, 0)
        print(f"  {rob:18} {n:>7}  {pct(n, total)}  {bar(n, total)}")

    # Per Label × Robustheit
    print(f"\n  Kreuztabelle Label × Robustheit:")
    print(f"  {'':8} {'validated':>10} {'single_src':>10} {'contested':>10} {'Total':>7}")
    print(f"  {'─' * 47}")
    cross = {}
    for label in LABELS:
        lt = [t for t in labeled_tracks if t.get("label") == label]
        robs = Counter(t.get("robustness", "single_source") for t in lt)
        n_val = robs.get("validated", 0)
        n_ss  = robs.get("single_source", 0)
        n_con = robs.get("contested", 0)
        n_tot = len(lt)
        val_pct_str = f"({pct(n_val, n_tot)})" if n_tot > 0 else ""
        print(f"  {label:8} {n_val:>7} {val_pct_str:>4} {n_ss:>10} {n_con:>10} {n_tot:>7}")
        cross[label] = {"validated": n_val, "single_source": n_ss, "contested": n_con}

    validated_pct = 100.0 * rob_counts.get("validated", 0) / total if total > 0 else 0
    if validated_pct < min_validated_pct:
        info(f"Nur {validated_pct:.0f}% validated — Last.fm-Enrichment verbessert die Qualität")

    return {"totals": dict(rob_counts), "per_label": cross}


# ══════════════════════════════════════════════════════════════════════════════
# SEKTION 4: TRAINING-QUALITÄT
# ══════════════════════════════════════════════════════════════════════════════

def section_training_quality(
    labeled_tracks: list[dict],
    validated_only: bool,
    min_flop_pct: float,
) -> dict:
    print(f"\n{'─' * 79}")
    print(f"  4. TRAINING-QUALITÄT")
    print(f"{'─' * 79}")

    weight_map = {"validated": 1.0, "single_source": 0.5, "contested": 0.7}

    weighted_counts = defaultdict(float)
    for t in labeled_tracks:
        label = t["label"]
        rob = t.get("robustness", "single_source")
        w = weight_map.get(rob, 0.5)
        weighted_counts[label] += w

    total_weight = sum(weighted_counts.values())

    print(f"  Effektives Gewicht je Label (nach Sample-Weights aus Robustheit):")
    print(f"  {'Label':8} {'Raw N':>7}  {'Gew. N':>8}  {'Gew. %':>7}")
    print(f"  {'─' * 40}")
    raw_counts = Counter(t["label"] for t in labeled_tracks)
    for label in LABELS:
        n_raw = raw_counts.get(label, 0)
        w_eff = weighted_counts.get(label, 0.0)
        print(f"  {label:8} {n_raw:>7}  {w_eff:>8.1f}  {pct(int(w_eff), int(total_weight))}")

    effective_imbalance_ratio = None
    if weighted_counts.get("flop", 0) > 0:
        max_lbl = max(weighted_counts, key=weighted_counts.get)
        ratio = weighted_counts[max_lbl] / weighted_counts["flop"]
        effective_imbalance_ratio = round(ratio, 2)
        print(f"\n  Effektive Imbalance (nach Gewichtung):")
        print(f"  {max_lbl.capitalize()} hat {ratio:.1f}× mehr Gewicht als Flop")
        if ratio > 3.0:
            warn(f"Selbst mit Gewichtung bleibt {max_lbl} {ratio:.1f}× dominanter")
            info("Balancierter Subset könnte Flop Recall verbessern")

    validated = [t for t in labeled_tracks if t.get("robustness") == "validated"]
    v_counts = Counter(t["label"] for t in validated)
    v_total = len(validated)

    print(f"\n  Validated-only Subset (n={v_total}):")
    print(f"  {'Label':8} {'Anzahl':>7}  {'Anteil':>7}  Balken")
    print(f"  {'─' * 45}")
    for label in LABELS:
        n = v_counts.get(label, 0)
        print(f"  {label:8} {n:>7}  {pct(n, v_total)}  {bar(n, v_total)}")

    if v_total > 0:
        v_flop_pct = 100.0 * v_counts.get("flop", 0) / v_total
        if v_flop_pct < min_flop_pct:
            warn(f"Auch in validated-only: Flop {v_flop_pct:.1f}% — Imbalance ist strukturell")
        print(f"\n  Validated Imbalance: {imbalance_ratio(v_counts)}")

    return {
        "weighted_counts": {k: round(v, 1) for k, v in weighted_counts.items()},
        "effective_imbalance_ratio": effective_imbalance_ratio,
        "validated_only": {
            "n": v_total,
            "labels": dict(v_counts),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# SEKTION 5: PER-CLUSTER
# ══════════════════════════════════════════════════════════════════════════════

def section_per_cluster(
    labeled_tracks: list[dict],
    min_flop_pct: float,
    focus_cluster: str = None,
) -> dict:
    print(f"\n{'─' * 79}")
    if focus_cluster:
        print(f"  5. PER-CLUSTER  (Fokus: {focus_cluster})")
    else:
        print(f"  5. PER-CLUSTER  (sortiert: Flop% absteigend)")
    print(f"{'─' * 79}")

    # Tracks nach Cluster gruppieren
    cluster_tracks = defaultdict(list)
    for t in labeled_tracks:
        for c in t.get("clusters", []):
            cluster_tracks[c].append(t)

    if not cluster_tracks:
        info("Keine Cluster-Informationen in den Daten.")
        return

    # Statistiken je Cluster berechnen
    cluster_stats = {}
    for cluster, tracks in cluster_tracks.items():
        counts = Counter(t["label"] for t in tracks)
        n = len(tracks)
        validated = sum(1 for t in tracks if t.get("robustness") == "validated")
        flop_pct = 100.0 * counts.get("flop", 0) / n if n > 0 else 0.0
        cluster_stats[cluster] = {
            "n": n,
            "hit": counts.get("hit", 0),
            "mid": counts.get("mid", 0),
            "flop": counts.get("flop", 0),
            "validated": validated,
            "flop_pct": flop_pct,
        }

    if focus_cluster:
        # Detailansicht für einen Cluster
        if focus_cluster not in cluster_stats:
            print(f"  Cluster '{focus_cluster}' nicht gefunden.")
            known = sorted(cluster_stats.keys())
            print(f"  Bekannte Cluster: {', '.join(known)}")
            return

        s = cluster_stats[focus_cluster]
        n = s["n"]
        print(f"  Cluster:      {focus_cluster}")
        print(f"  Tracks:       {n}")
        print(f"  Validated:    {s['validated']} ({pct(s['validated'], n)})")
        print()
        for label in LABELS:
            cnt = s[label]
            print(f"  {label:8} {cnt:>6}  ({pct(cnt, n)})  {bar(cnt, n)}")

        if s["flop_pct"] < min_flop_pct:
            warn(f"Flop-Anteil {s['flop_pct']:.1f}% — Cluster wenig hilfreich für Flop Recall")
        else:
            info(f"Flop-Anteil {s['flop_pct']:.1f}% — Cluster nützlich für Flop Recall")

        cluster_t = cluster_tracks[focus_cluster]
        rob_counts = Counter(t.get("robustness", "single_source") for t in cluster_t)
        print(f"\n  Robustheit:")
        for rob in ROBUSTNESS_LEVELS:
            cnt = rob_counts.get(rob, 0)
            print(f"  {rob:18} {cnt:>6}  ({pct(cnt, n)})")

    else:
        sorted_clusters = sorted(
            cluster_stats.items(),
            key=lambda x: x[1]["flop_pct"],
            reverse=True,
        )

        print(f"  {'Cluster':22} {'N':>5}  {'Hit%':>6} {'Mid%':>6} {'Flop%':>6} {'Val%':>6}  Hinweis")
        print(f"  {'─' * 72}")

        for cluster, s in sorted_clusters:
            n = s["n"]
            hit_p  = 100.0 * s["hit"]  / n if n > 0 else 0
            mid_p  = 100.0 * s["mid"]  / n if n > 0 else 0
            flop_p = s["flop_pct"]
            val_p  = 100.0 * s["validated"] / n if n > 0 else 0

            hint = ""
            if flop_p >= 25:
                hint = "★ Flop-reich"
            elif flop_p < min_flop_pct:
                hint = "! wenig Flops"
            elif hit_p >= 60:
                hint = "· Hit-lastig"

            print(f"  {cluster:22} {n:>5}  {hit_p:>5.1f}% {mid_p:>5.1f}% {flop_p:>5.1f}% {val_p:>5.1f}%  {hint}")

        flop_rich = sum(1 for _, s in cluster_stats.items() if s["flop_pct"] >= 25)
        flop_poor = sum(1 for _, s in cluster_stats.items() if s["flop_pct"] < min_flop_pct)
        print(f"\n  ★ Flop-reiche Cluster (≥25%): {flop_rich}")
        print(f"  ! Flop-arme Cluster (<{min_flop_pct:.0f}%):  {flop_poor}")

    # Report-Daten: kompaktes Dict je Cluster
    report_data = {
        cluster: {
            "n": s["n"],
            "hit": s["hit"],
            "mid": s["mid"],
            "flop": s["flop"],
            "validated": s["validated"],
            "flop_pct": round(s["flop_pct"], 1),
        }
        for cluster, s in cluster_stats.items()
    }
    return report_data


# ══════════════════════════════════════════════════════════════════════════════
# SEKTION 6: OVERLAP
# ══════════════════════════════════════════════════════════════════════════════

def section_overlap(tracks: list[dict]) -> dict:
    print(f"\n{'─' * 79}")
    print(f"  6. OVERLAP  (Tracks in mehreren Clustern)")
    print(f"{'─' * 79}")

    cluster_counts = Counter(len(t.get("clusters", [])) for t in tracks)
    total = len(tracks)

    print(f"  {'Cluster-Zuord.':18} {'Tracks':>7}  {'Anteil':>7}")
    print(f"  {'─' * 38}")
    for n_clusters in sorted(cluster_counts.keys()):
        cnt = cluster_counts[n_clusters]
        label = f"{n_clusters} Cluster" if n_clusters != 1 else "1 Cluster (exklusiv)"
        print(f"  {label:18} {cnt:>7}  {pct(cnt, total)}")

    multi = sum(v for k, v in cluster_counts.items() if k > 1)
    if multi > 0:
        info(
            f"{multi} Tracks ({pct(multi, total)}) in mehreren Clustern — "
            f"kein Doppelzählen im Training (track_id ist Primärschlüssel)"
        )

    return {str(k): v for k, v in sorted(cluster_counts.items())}


# ══════════════════════════════════════════════════════════════════════════════
# SEKTION 7: EMBEDDING-STATUS
# ══════════════════════════════════════════════════════════════════════════════

def section_embedding_status(labeled_tracks: list[dict], embeddings_base: Path) -> dict:
    print(f"\n{'─' * 79}")
    print(f"  7. EMBEDDING-STATUS")
    print(f"{'─' * 79}")

    meta_files = list(embeddings_base.glob("**/embeddings_meta.csv"))

    if not meta_files:
        info(f"Keine embeddings_meta.csv gefunden in {embeddings_base}")
        info("→ extract_embeddings.py noch nicht gelaufen")
        return {}

    import csv
    labeled_ids = {t["track_id"] for t in labeled_tracks}
    result = {}

    for meta_path in sorted(meta_files):
        embedder_dir = meta_path.parent.name
        try:
            with open(meta_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                embedded_ids = {int(row["track_id"]) for row in reader if row.get("track_id")}
        except Exception as e:
            print(f"  [{embedder_dir}] Fehler beim Lesen: {e}")
            continue

        n_embedded = len(embedded_ids)
        overlap = labeled_ids & embedded_ids
        n_overlap = len(overlap)
        n_labeled_not_embedded = len(labeled_ids - embedded_ids)
        n_embedded_not_labeled = len(embedded_ids - labeled_ids)

        trainable = [t for t in labeled_tracks if t["track_id"] in overlap]
        tc = Counter(t["label"] for t in trainable)
        total = len(trainable)

        print(f"  Embedder: {embedder_dir}")
        print(f"    Embeddings vorhanden:        {n_embedded:>6}")
        print(f"    Gelabelt + Embedding:        {n_overlap:>6}  ← tatsächliches Trainingsset")
        print(f"    Gelabelt aber kein Embedding:{n_labeled_not_embedded:>6}  (extract_embeddings nötig)")
        print(f"    Embedding aber kein Label:   {n_embedded_not_labeled:>6}  (Labels berechnen nötig)")
        print(f"\n    Trainingsset Label-Verteilung:")
        for label in LABELS:
            n = tc.get(label, 0)
            print(f"      {label:8} {n:>6}  ({pct(n, total)})  {bar(n, total, width=15)}")
        print()

        result[embedder_dir] = {
            "n_embedded": n_embedded,
            "n_trainable": n_overlap,
            "n_labeled_no_embedding": n_labeled_not_embedded,
            "n_embedded_no_label": n_embedded_not_labeled,
            "trainable_labels": dict(tc),
        }

    return result


# ══════════════════════════════════════════════════════════════════════════════
# FAZIT
# ══════════════════════════════════════════════════════════════════════════════

def section_fazit(
    label_counts: Counter,
    rob_counts: dict,
    min_flop_pct: float,
) -> dict:
    print(f"\n{'═' * 79}")
    print(f"  FAZIT")
    print(f"{'═' * 79}")

    total = sum(label_counts.values())
    flop_n = label_counts.get("flop", 0)
    hit_n  = label_counts.get("hit", 0)
    mid_n  = label_counts.get("mid", 0)
    rob_totals = rob_counts.get("totals", rob_counts)  # Kompatibel mit altem Counter
    val_n  = rob_totals.get("validated", 0)

    flop_pct  = 100.0 * flop_n / total if total > 0 else 0
    val_pct   = 100.0 * val_n / total if total > 0 else 0

    issues = []
    suggestions = []

    if flop_pct < min_flop_pct:
        issues.append(f"Flop-Anteil {flop_pct:.1f}% (Ziel: ≥{min_flop_pct:.0f}%)")
        if flop_n < 100:
            suggestions.append("Mehr Flop-Daten scouten (flop-reiche Cluster priorisieren)")
        suggestions.append("Balanced-Subset erstellen (balance_dataset.py)")

    if val_pct < DEFAULT_MIN_VALIDATED_PCT:
        issues.append(f"Validated-Anteil {val_pct:.1f}% (Ziel: ≥{DEFAULT_MIN_VALIDATED_PCT:.0f}%)")
        suggestions.append("Last.fm-Enrichment für mehr validated-Labels nachholen")

    if hit_n > 0 and mid_n > 0 and flop_n > 0:
        ratio = max(hit_n, mid_n) / flop_n
        if ratio > 5:
            issues.append(f"Imbalance-Ratio {ratio:.1f}× — Modell wird Flop ignorieren")
            suggestions.append(f"Balanced-Subset: ~{flop_n} Hit, ~{min(mid_n, flop_n*2)} Mid, {flop_n} Flop")

    if issues:
        print(f"\n  Probleme:")
        for issue in issues:
            print(f"    ✗  {issue}")
        print(f"\n  Empfehlungen:")
        for sug in suggestions:
            print(f"    →  {sug}")
    else:
        print(f"\n  ✓  Datenbasis sieht trainierbar aus.")
        print(f"     Flop: {flop_pct:.1f}%  |  Validated: {val_pct:.1f}%  |  Imbalance: {imbalance_ratio(label_counts)}")

    print(f"{'═' * 79}")

    return {"issues": issues, "suggestions": suggestions}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def write_report(report: dict, reports_dir: Path) -> Path:
    """Schreibt den Inspection-Report als JSON."""
    ensure_dir(reports_dir)
    date_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"dataset_inspection_{date_tag}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose-Tool für Trainingsdaten (read-only, optional mit Report)"
    )
    parser.add_argument(
        "--cluster",
        metavar="NAME",
        default=None,
        help="Detailansicht für einen bestimmten Cluster",
    )
    parser.add_argument(
        "--min-flop-pct",
        type=float,
        default=DEFAULT_MIN_FLOP_PCT,
        metavar="PCT",
        help=f"Mindest-Flop-Anteil in %% für Warnungen (default: {DEFAULT_MIN_FLOP_PCT})",
    )
    parser.add_argument(
        "--validated-only",
        action="store_true",
        help="Sektion 4 basiert nur auf validated-Tracks",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Sektion 7 (Embedding-Status) überspringen",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Report als JSON in outputs/reports/ speichern",
    )
    args = parser.parse_args()

    # Pfade laden
    paths = load_paths_config()
    metadata_dir = paths.get("metadata")
    jsonl_path = get_tracks_jsonl_path(metadata_dir)
    embeddings_base = paths.get("embeddings", Path("./outputs/embeddings"))
    reports_dir = paths.get("reports", Path("./outputs/reports"))

    # ── Header ──
    print(f"{'═' * 79}")
    print(f"  DATASET INSPECTION")
    print(f"{'═' * 79}")
    print(f"  Quelle: {jsonl_path}")

    if not jsonl_path.exists():
        print(f"\n  Fehler: tracks.jsonl nicht gefunden: {jsonl_path}")
        print(f"  → Erst 'python scripts/scout_deezer.py' ausführen!")
        sys.exit(1)

    # Tracks laden
    all_tracks = read_tracks(jsonl_path)
    labeled_tracks = [t for t in all_tracks if t.get("label") is not None]

    print(f"  Geladen: {len(all_tracks)} Tracks ({len(labeled_tracks)} gelabelt)")

    # ── Sektionen ──
    r_pipeline = section_pipeline_status(all_tracks, min_labeled_pct=DEFAULT_MIN_LABELED_PCT)

    if not labeled_tracks:
        print(f"\n  Keine gelabelten Tracks — Sektionen 2-7 übersprungen.")
        print(f"  → compute_labels.py ausführen!")
        sys.exit(0)

    label_counts  = section_label_distribution(labeled_tracks, args.min_flop_pct)
    r_robustness  = section_robustness(labeled_tracks, DEFAULT_MIN_VALIDATED_PCT)
    r_quality     = section_training_quality(labeled_tracks, args.validated_only, args.min_flop_pct)
    r_clusters    = section_per_cluster(labeled_tracks, args.min_flop_pct, focus_cluster=args.cluster)
    r_overlap     = section_overlap(all_tracks)
    r_embeddings  = {} if args.skip_embeddings else section_embedding_status(labeled_tracks, embeddings_base)
    r_fazit       = section_fazit(label_counts, r_robustness, args.min_flop_pct)

    # ── Report schreiben ──
    if args.report:
        report = {
            "created": datetime.now().isoformat(),
            "source": str(jsonl_path),
            "min_flop_pct_threshold": args.min_flop_pct,
            "pipeline_status": r_pipeline,
            "label_distribution": dict(label_counts),
            "robustness": r_robustness,
            "training_quality": r_quality,
            "clusters": r_clusters,
            "overlap": r_overlap,
            "embeddings": r_embeddings,
            "fazit": r_fazit,
        }
        report_path = write_report(report, reports_dir)
        print(f"\n  Report gespeichert: {report_path}")


if __name__ == "__main__":
    main()
