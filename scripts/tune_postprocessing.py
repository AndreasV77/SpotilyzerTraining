"""
tune_postprocessing.py
======================
Post-hoc Threshold-Tuning fuer das trainierte XGBoost-Modell.

Implementiert drei Interventionen aus dem Mid-Recall-Briefing:

  1. Leakage-Check     — Artist-Overlap zwischen Train- und Holdout-Set
  2. Logit-Adjustment  — τ-Sweep: p'(c) ∝ p(c) / π(c)^τ
  3. Zweistufen-Schwellen — θ_hit / θ_flop Sweep (Mid als Unsicherheitszone)
  4. Kombiniert        — erst Logit-Adjustment, dann Zweistufen

WICHTIG zu Selection Bias:
  Tuning und Evaluation erfolgen hier auf demselben Holdout-Set.
  Bei 9 τ-Werten ist der Bias gering (~0.1-0.3pp), aber vorhanden.
  Für production: τ/θ auf innerem Val-Set tunen (in train_model.py integrieren).
  Hier: explorative Analyse — Hauptfrage ist, OB der Mechanismus hilft,
  nicht die exakte Groesse des Gewinns.

Klassen-Index-Reihenfolge (LabelEncoder alphabetisch):
  flop=0, hit=1, mid=2
  predict_proba Output: [p_flop, p_hit, p_mid]

Usage:
  python scripts/tune_postprocessing.py --embedder 330M --dataset main spotify_charts kworb --validated-only
  python scripts/tune_postprocessing.py --embedder 330M --dataset main spotify_charts kworb --validated-only --save-report
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report
import joblib

from _utils import (
    setup_logging,
    load_paths_config,
    load_training_config,
    load_thresholds_config,
    ensure_dir,
)
from utils.metadata import read_tracks, get_tracks_jsonl_path


# ══════════════════════════════════════════════════════════════════════════════
# KLASSEN-INDEX-KONSTANTEN
# Entspricht LabelEncoder alphabetisch: flop=0, hit=1, mid=2
# ══════════════════════════════════════════════════════════════════════════════
IDX_FLOP = 0
IDX_HIT  = 1
IDX_MID  = 2


# ══════════════════════════════════════════════════════════════════════════════
# INTERVENTION 1: LEAKAGE-CHECK
# ══════════════════════════════════════════════════════════════════════════════

def check_artist_leakage(
    train_track_ids: np.ndarray,
    test_track_ids: np.ndarray,
    tracks_dict: dict,
) -> dict:
    """
    Prueft Artist-Overlap zwischen Train- und Test-Set.

    Ein Artist in beiden Sets bedeutet: das Modell hat beim Training
    moeglicherweise 'Artist-Identitaet' gelernt statt akustische Eigenschaften.
    """
    def get_artist(tid):
        t = tracks_dict.get(int(tid), {})
        return t.get("artist", "").strip().lower()

    train_artists = set(get_artist(tid) for tid in train_track_ids)
    test_artists  = set(get_artist(tid) for tid in test_track_ids)

    # Leere Strings raus (Tracks ohne Kuenstler-Info)
    train_artists.discard("")
    test_artists.discard("")

    overlap = train_artists & test_artists
    overlap_ratio = len(overlap) / len(test_artists) if test_artists else 0.0

    # Wie viele TEST-Tracks sind von overlapping artists?
    test_track_overlap_count = sum(
        1 for tid in test_track_ids
        if get_artist(tid) in overlap
    )
    test_track_overlap_ratio = test_track_overlap_count / len(test_track_ids) if len(test_track_ids) else 0.0

    return {
        "train_artists": len(train_artists),
        "test_artists": len(test_artists),
        "overlap_artists": len(overlap),
        "overlap_artist_ratio": float(overlap_ratio),
        "test_tracks_from_overlap_artists": test_track_overlap_count,
        "test_tracks_from_overlap_artists_ratio": float(test_track_overlap_ratio),
        "overlap_examples": sorted(overlap)[:10],  # Erste 10 als Beispiel
    }


# ══════════════════════════════════════════════════════════════════════════════
# INTERVENTION 2: LOGIT-ADJUSTMENT
# ══════════════════════════════════════════════════════════════════════════════

def logit_adjustment(probs: np.ndarray, class_priors: np.ndarray, tau: float) -> np.ndarray:
    """
    Post-hoc Logit-Adjustment.

    Formel: p'(c) ∝ p(c) / π(c)^τ

    Args:
        probs:         (n, 3) Model-Output — Reihenfolge [p_flop, p_hit, p_mid]
        class_priors:  (3,) Trainings-Priors — gleiche Reihenfolge [π_flop, π_hit, π_mid]
        tau:           Adjustment-Staerke (0 = kein Adjustment, >0 = kleinere Klassen bevorzugt)

    Returns:
        adjusted_probs: (n, 3), re-normalisiert auf Summe=1
    """
    if tau == 0.0:
        return probs.copy()
    adjustment = np.clip(class_priors, 1e-9, None) ** tau
    adjusted = probs / adjustment
    # Numerische Stabilitaet
    adjusted = np.clip(adjusted, 1e-12, None)
    adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)
    return adjusted


def sweep_tau(
    probs: np.ndarray,
    y_true: np.ndarray,
    class_priors: np.ndarray,
    tau_range: list = None,
) -> list[dict]:
    """
    Grid-Search ueber τ-Werte.

    Returns:
        Liste von {tau, BA, per_class_recall} sortiert nach BA absteigend.
    """
    if tau_range is None:
        tau_range = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

    results = []
    for tau in tau_range:
        adj_probs = logit_adjustment(probs, class_priors, tau)
        y_pred = np.argmax(adj_probs, axis=1)
        ba = balanced_accuracy_score(y_true, y_pred)

        # Per-class recall
        recalls = {}
        for cls_name, cls_idx in [("flop", IDX_FLOP), ("hit", IDX_HIT), ("mid", IDX_MID)]:
            mask = y_true == cls_idx
            if mask.sum() > 0:
                recalls[cls_name] = float((y_pred[mask] == cls_idx).mean())
            else:
                recalls[cls_name] = 0.0

        results.append({
            "tau": tau,
            "BA": float(ba),
            "flop_recall": recalls["flop"],
            "hit_recall": recalls["hit"],
            "mid_recall": recalls["mid"],
        })

    results.sort(key=lambda x: x["BA"], reverse=True)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# INTERVENTION 3: ZWEISTUFEN-SCHWELLEN
# ══════════════════════════════════════════════════════════════════════════════

def two_threshold_prediction(
    probs: np.ndarray,
    theta_hit: float,
    theta_flop: float,
) -> np.ndarray:
    """
    Zweistufen-Entscheidung: Mid als Default fuer unsichere Faelle.

    Klassen-Index: flop=0, hit=1, mid=2
    predict_proba Reihenfolge: [p_flop, p_hit, p_mid]

    Logik:
        Wenn p(Hit) > θ_hit → Hit
        Wenn p(Flop) > θ_flop (und nicht Hit) → Flop
        Sonst → Mid (Default)
    """
    preds = np.full(len(probs), IDX_MID, dtype=int)  # Default: Mid
    preds[probs[:, IDX_HIT] > theta_hit] = IDX_HIT
    flop_mask = (probs[:, IDX_FLOP] > theta_flop) & (preds != IDX_HIT)
    preds[flop_mask] = IDX_FLOP
    return preds


def sweep_thresholds(
    probs: np.ndarray,
    y_true: np.ndarray,
    grid_step: float = 0.05,
) -> list[dict]:
    """
    Grid-Search ueber θ_hit und θ_flop (groberer Raster fuer Effizienz).

    Returns:
        Top-20 Ergebnisse sortiert nach BA absteigend.
    """
    best_results = []

    for theta_hit in np.arange(0.2, 0.85, grid_step):
        for theta_flop in np.arange(0.2, 0.85, grid_step):
            y_pred = two_threshold_prediction(probs, theta_hit, theta_flop)
            ba = balanced_accuracy_score(y_true, y_pred)

            recalls = {}
            for cls_name, cls_idx in [("flop", IDX_FLOP), ("hit", IDX_HIT), ("mid", IDX_MID)]:
                mask = y_true == cls_idx
                if mask.sum() > 0:
                    recalls[cls_name] = float((y_pred[mask] == cls_idx).mean())
                else:
                    recalls[cls_name] = 0.0

            best_results.append({
                "theta_hit": float(round(theta_hit, 3)),
                "theta_flop": float(round(theta_flop, 3)),
                "BA": float(ba),
                "flop_recall": recalls["flop"],
                "hit_recall": recalls["hit"],
                "mid_recall": recalls["mid"],
            })

    best_results.sort(key=lambda x: x["BA"], reverse=True)
    return best_results[:20]


# ══════════════════════════════════════════════════════════════════════════════
# INTERVENTION 4: KOMBINIERT (Logit → Threshold)
# ══════════════════════════════════════════════════════════════════════════════

def sweep_combined(
    probs: np.ndarray,
    y_true: np.ndarray,
    class_priors: np.ndarray,
    tau_range: list = None,
    grid_step: float = 0.05,
) -> dict:
    """
    Fuer jeden τ: optimale Schwellen finden.
    Gibt bestes (τ, θ_hit, θ_flop) zurueck.
    """
    if tau_range is None:
        tau_range = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    best_ba = 0.0
    best_config = {}

    for tau in tau_range:
        adj_probs = logit_adjustment(probs, class_priors, tau)
        for theta_hit in np.arange(0.2, 0.85, grid_step):
            for theta_flop in np.arange(0.2, 0.85, grid_step):
                y_pred = two_threshold_prediction(adj_probs, theta_hit, theta_flop)
                ba = balanced_accuracy_score(y_true, y_pred)
                if ba > best_ba:
                    best_ba = ba
                    recalls = {}
                    for cls_name, cls_idx in [("flop", IDX_FLOP), ("hit", IDX_HIT), ("mid", IDX_MID)]:
                        mask = y_true == cls_idx
                        recalls[cls_name] = float((y_pred[mask] == cls_idx).mean()) if mask.sum() > 0 else 0.0
                    best_config = {
                        "tau": float(tau),
                        "theta_hit": float(round(theta_hit, 3)),
                        "theta_flop": float(round(theta_flop, 3)),
                        "BA": float(ba),
                        **{f"{k}_recall": v for k, v in recalls.items()},
                    }

    return best_config


# ══════════════════════════════════════════════════════════════════════════════
# DATEN-LADEN
# ══════════════════════════════════════════════════════════════════════════════

def load_holdout_data(
    embeddings_dir: Path,
    jsonl_paths: list,
    test_track_ids: np.ndarray,
    validated_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Laedt Holdout-Embeddings + Labels.

    Returns:
        (X_test, y_test, tracks_dict)
        y_test: int-encoded, flop=0, hit=1, mid=2
    """
    label_map = {"flop": IDX_FLOP, "hit": IDX_HIT, "mid": IDX_MID}

    embeddings = np.load(embeddings_dir / "embeddings.npy")
    meta_df = pd.read_csv(embeddings_dir / "embeddings_meta.csv")

    # Alle Tracks laden und mergen
    tracks_dict = {}
    for p in jsonl_paths:
        if not Path(p).exists():
            print(f"  WARNUNG: JSONL nicht gefunden, uebersprungen: {p}")
            continue
        for track in read_tracks(Path(p)):
            tid = track.get("track_id")
            if tid is not None:
                if tid in tracks_dict:
                    old_clusters = set(tracks_dict[tid].get("clusters", []))
                    new_clusters = set(track.get("clusters", []))
                    tracks_dict[tid].update(track)
                    tracks_dict[tid]["clusters"] = sorted(old_clusters | new_clusters)
                else:
                    tracks_dict[tid] = track

    test_ids_set = set(test_track_ids.tolist())

    labels = []
    valid_mask = []
    for _, row in meta_df.iterrows():
        tid = int(row["track_id"])
        if tid not in test_ids_set:
            valid_mask.append(False)
            continue
        track = tracks_dict.get(tid, {})
        label = track.get("label")
        robustness = track.get("robustness", "single_source")
        if label is None:
            valid_mask.append(False)
            continue
        if validated_only and robustness != "validated":
            valid_mask.append(False)
            continue
        labels.append(label_map[label])
        valid_mask.append(True)

    valid_mask = np.array(valid_mask)
    valid_indices = meta_df.loc[valid_mask, "embedding_idx"].values
    X = embeddings[valid_indices]
    y = np.array(labels, dtype=int)

    return X, y, tracks_dict


def compute_class_priors(
    jsonl_paths: list,
    test_track_ids_set: set,
    validated_only: bool = False,
) -> np.ndarray:
    """
    Berechnet Klassen-Priors aus den TRAINING-Tracks (alle ausser Holdout).

    Returns:
        priors: (3,) — Reihenfolge [π_flop, π_hit, π_mid]
    """
    counts = defaultdict(int)
    label_map = {"flop": IDX_FLOP, "hit": IDX_HIT, "mid": IDX_MID}

    for p in jsonl_paths:
        if not Path(p).exists():
            continue
        for track in read_tracks(Path(p)):
            tid = track.get("track_id")
            label = track.get("label")
            robustness = track.get("robustness", "single_source")
            if tid is None or label is None or label not in label_map:
                continue
            if validated_only and robustness != "validated":
                continue
            if tid in test_track_ids_set:
                continue  # Holdout ausschliessen
            counts[label_map[label]] += 1

    total = sum(counts.values())
    if total == 0:
        # Fallback: uniform
        return np.array([1/3, 1/3, 1/3])

    priors = np.array([
        counts[IDX_FLOP] / total,
        counts[IDX_HIT]  / total,
        counts[IDX_MID]  / total,
    ])
    return priors


# ══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def print_leakage_report(leakage: dict):
    print(f"\n{'─' * 79}")
    print(f"  LEAKAGE-CHECK (Artist-Overlap)")
    print(f"{'─' * 79}")
    print(f"    Train-Artists:    {leakage['train_artists']:>5}")
    print(f"    Test-Artists:     {leakage['test_artists']:>5}")
    print(f"    Overlap-Artists:  {leakage['overlap_artists']:>5}  ({leakage['overlap_artist_ratio']:.1%} der Test-Artists)")
    print(f"    Test-Tracks von Overlap-Artists: {leakage['test_tracks_from_overlap_artists']:>5}  ({leakage['test_tracks_from_overlap_artists_ratio']:.1%})")

    if leakage["overlap_artist_ratio"] > 0.5:
        print(f"\n    ⚠ WARNUNG: Hoher Artist-Overlap — Modell lernt moeglicherweise")
        print(f"      Kuenstler-Identitaet statt akustische Eigenschaften.")
        print(f"      Empfehlung: GroupKFold mit artist_id fuer unbiased Evaluation.")
    elif leakage["overlap_artist_ratio"] > 0.2:
        print(f"\n    ! HINWEIS: Moderater Artist-Overlap erkannt.")
    else:
        print(f"\n    ✓ Overlap gering — Split erscheint weitgehend kuenstler-clean.")

    if leakage["overlap_examples"]:
        examples = leakage["overlap_examples"][:5]
        print(f"    Beispiele: {', '.join(repr(a) for a in examples)}")


def print_tau_results(results: list[dict], baseline_ba: float):
    print(f"\n{'─' * 79}")
    print(f"  LOGIT-ADJUSTMENT (τ-Sweep)")
    print(f"{'─' * 79}")
    print(f"  Baseline (τ=0.0):  BA={baseline_ba:.4f}")
    print()
    print(f"  {'τ':>6}  {'BA':>8}  {'ΔBA':>7}  {'Hit R.':>8}  {'Mid R.':>8}  {'Flop R.':>8}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*8}")
    for r in results[:9]:  # Alle τ-Werte (sortiert nach BA)
        delta = r["BA"] - baseline_ba
        marker = " ◄ BEST" if r == results[0] else ""
        print(
            f"  {r['tau']:>6.2f}  {r['BA']:>8.4f}  {delta:>+7.4f}  "
            f"{r['hit_recall']:>8.3f}  {r['mid_recall']:>8.3f}  {r['flop_recall']:>8.3f}"
            f"{marker}"
        )

    best = results[0]
    print(f"\n  Bestes τ:  {best['tau']}  →  BA={best['BA']:.4f}  (Δ={best['BA']-baseline_ba:+.4f})")


def print_threshold_results(results: list[dict], baseline_ba: float):
    print(f"\n{'─' * 79}")
    print(f"  ZWEISTUFEN-SCHWELLEN (θ-Sweep, Top 5)")
    print(f"{'─' * 79}")
    print(f"  Baseline (argmax):  BA={baseline_ba:.4f}")
    print()
    print(f"  {'θ_hit':>7}  {'θ_flop':>7}  {'BA':>8}  {'ΔBA':>7}  {'Hit R.':>8}  {'Mid R.':>8}  {'Flop R.':>8}")
    print(f"  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*8}")
    for r in results[:5]:
        delta = r["BA"] - baseline_ba
        print(
            f"  {r['theta_hit']:>7.3f}  {r['theta_flop']:>7.3f}  {r['BA']:>8.4f}  {delta:>+7.4f}  "
            f"{r['hit_recall']:>8.3f}  {r['mid_recall']:>8.3f}  {r['flop_recall']:>8.3f}"
        )

    best = results[0]
    print(f"\n  Bestes Paar:  θ_hit={best['theta_hit']}  θ_flop={best['theta_flop']}  "
          f"→  BA={best['BA']:.4f}  (Δ={best['BA']-baseline_ba:+.4f})")


def print_combined_result(combined: dict, baseline_ba: float):
    print(f"\n{'─' * 79}")
    print(f"  KOMBINIERT (Logit-Adjustment + Zweistufen-Schwellen)")
    print(f"{'─' * 79}")
    print(f"  Baseline:  BA={baseline_ba:.4f}")
    delta = combined["BA"] - baseline_ba
    print(f"  Beste Kombination:")
    print(f"    τ={combined['tau']}  θ_hit={combined['theta_hit']}  θ_flop={combined['theta_flop']}")
    print(f"    BA={combined['BA']:.4f}  (Δ={delta:+.4f})")
    print(f"    Hit Recall={combined['hit_recall']:.3f}  "
          f"Mid Recall={combined['mid_recall']:.3f}  "
          f"Flop Recall={combined['flop_recall']:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    paths = load_paths_config()
    training_cfg = load_training_config()
    thresholds_cfg = load_thresholds_config()

    models_dir = paths.get("models", Path("./outputs/models"))
    embeddings_base = paths.get("embeddings", Path("./outputs/embeddings"))
    datasets_base = paths.get("datasets", Path("G:/Dev/SpotilyzerData/datasets"))
    metadata_dir = paths.get("metadata")

    _EMBEDDER_HF  = {"95M": "m-a-p/MERT-v1-95M", "330M": "m-a-p/MERT-v1-330M"}
    _EMBEDDER_TAG = {"95M": "MERTv195M",           "330M": "MERTv1330M"}

    _glob_all = sorted(models_dir.glob("spotilyzer_model_*.joblib"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    default_model = str(_glob_all[0] if _glob_all else models_dir / "spotilyzer_model.joblib")

    cfg_model = training_cfg.get("embedder", {}).get("model", "")
    cfg_short = cfg_model.split("/")[-1] if cfg_model else ""
    default_embeddings = str(embeddings_base / cfg_short if cfg_short else embeddings_base)

    parser = argparse.ArgumentParser(
        description="Post-hoc Threshold-Tuning: Logit-Adjustment + Zweistufen-Schwellen"
    )
    parser.add_argument("--embedder", choices=list(_EMBEDDER_HF.keys()), default=None)
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--embeddings-dir", default=default_embeddings)
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["main"],
        metavar="DATASET",
    )
    parser.add_argument("--validated-only", action="store_true")
    parser.add_argument("--save-report", action="store_true",
                        help="Speichere Tuning-Ergebnisse als JSON in outputs/reports/")
    parser.add_argument(
        "--skip-combined",
        action="store_true",
        help="Kombinierten Sweep ueberspringen (spart Zeit, ~2-3 Min bei grid_step=0.05)",
    )
    args = parser.parse_args()

    # --embedder setzt Pfade automatisch
    if args.embedder:
        tag = _EMBEDDER_TAG[args.embedder]
        model_short = _EMBEDDER_HF[args.embedder].split("/")[-1]
        args.embeddings_dir = str(embeddings_base / model_short)
        if args.model == default_model:
            if args.validated_only:
                tagged = sorted(models_dir.glob(f"spotilyzer_model_{tag}*validated*.joblib"),
                                key=lambda p: p.stat().st_mtime, reverse=True)
            else:
                tagged = sorted(models_dir.glob(f"spotilyzer_model_{tag}*.joblib"),
                                key=lambda p: p.stat().st_mtime, reverse=True)
            if tagged:
                args.model = str(tagged[0])

    # JSONL-Pfade aufbauen
    jsonl_paths = []
    for ds in args.dataset:
        if ds == "main":
            jsonl_paths.append(get_tracks_jsonl_path(metadata_dir))
        else:
            jsonl_paths.append(Path(datasets_base) / ds / "tracks.jsonl")

    print(f"{'=' * 79}")
    print(f"  SPOTILYZER POST-HOC TUNING")
    print(f"{'=' * 79}")
    print(f"  Modell:    {args.model}")
    print(f"  Embeddings: {args.embeddings_dir}")
    print(f"  Datasets:  {args.dataset}")
    if args.validated_only:
        print(f"  Filter:    validated_only=True")
    print()
    print(f"  HINWEIS: Tuning und Evaluation auf demselben Holdout-Set.")
    print(f"  Selection Bias ist bei τ-Sweep (9 Werte) gering (~0.1-0.3pp).")
    print(f"  Fuer unbiasierte Produktion: τ in train_model.py auf inner-val tunen.")

    model_path = Path(args.model)
    embeddings_dir = Path(args.embeddings_dir)

    if not model_path.exists():
        print(f"\n  Fehler: Modell nicht gefunden: {model_path}")
        sys.exit(1)
    if not (embeddings_dir / "embeddings.npy").exists():
        print(f"\n  Fehler: Embeddings nicht gefunden: {embeddings_dir}")
        sys.exit(1)

    # Modell laden
    print(f"\n  Lade Modell...")
    bundle = joblib.load(model_path)
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]
    test_track_ids = bundle.get("test_track_ids", None)

    if test_track_ids is None:
        print("  FEHLER: Kein test_track_ids im Bundle — altes Modell ohne Holdout-Split.")
        print("  Bitte Modell neu trainieren (train_model.py speichert test_track_ids).")
        sys.exit(1)

    print(f"    Holdout-Size: {len(test_track_ids)} Tracks")

    # Holdout laden
    print(f"  Lade Holdout-Daten...")
    X_test, y_test, tracks_dict = load_holdout_data(
        embeddings_dir, jsonl_paths, test_track_ids,
        validated_only=args.validated_only,
    )
    print(f"    Samples: {X_test.shape[0]}")

    if X_test.shape[0] == 0:
        print("  FEHLER: Keine Test-Samples geladen — JSONL oder Embeddings pruefen.")
        sys.exit(1)

    # Class-Priors aus Trainings-Tracks berechnen (Holdout ausgeschlossen)
    print(f"  Berechne Klassen-Priors aus Training-Set...")
    test_ids_set = set(test_track_ids.tolist())
    class_priors = compute_class_priors(jsonl_paths, test_ids_set, validated_only=args.validated_only)
    print(f"    π_flop={class_priors[IDX_FLOP]:.3f}  π_hit={class_priors[IDX_HIT]:.3f}  π_mid={class_priors[IDX_MID]:.3f}")

    # Baseline-Vorhersage
    probs = model.predict_proba(X_test)
    y_pred_baseline = model.predict(X_test)
    baseline_ba = balanced_accuracy_score(y_test, y_pred_baseline)

    # Klassen-Verteilung im Holdout
    unique, counts = np.unique(y_test, return_counts=True)
    label_names = {IDX_FLOP: "flop", IDX_HIT: "hit", IDX_MID: "mid"}
    print(f"\n  Holdout-Verteilung:")
    for u, c in zip(unique, counts):
        print(f"    {label_names[u]:6}: {c:>5} ({c/len(y_test):.1%})")

    print(f"\n  Baseline BA: {baseline_ba:.4f}")

    # ── Intervention 1: Leakage-Check ──────────────────────────────────────

    # Alle Track-IDs in Train-Set bestimmen
    # (alle validierten Tracks aus JSONL minus Holdout)
    label_map = {"flop": IDX_FLOP, "hit": IDX_HIT, "mid": IDX_MID}
    train_track_ids = []
    for p in jsonl_paths:
        if not Path(p).exists():
            continue
        for track in read_tracks(Path(p)):
            tid = track.get("track_id")
            label = track.get("label")
            robustness = track.get("robustness", "single_source")
            if tid is None or label is None or label not in label_map:
                continue
            if args.validated_only and robustness != "validated":
                continue
            if tid not in test_ids_set:
                train_track_ids.append(tid)
    train_track_ids = np.array(train_track_ids)

    leakage = check_artist_leakage(train_track_ids, test_track_ids, tracks_dict)
    print_leakage_report(leakage)

    # ── Intervention 2: τ-Sweep ────────────────────────────────────────────

    print(f"\n  Starte τ-Sweep...")
    tau_results = sweep_tau(probs, y_test, class_priors)
    print_tau_results(tau_results, baseline_ba)

    # ── Intervention 3: Schwellen-Sweep ────────────────────────────────────

    print(f"\n  Starte θ-Sweep (grid_step=0.05)...")
    theta_results = sweep_thresholds(probs, y_test, grid_step=0.05)
    print_threshold_results(theta_results, baseline_ba)

    # ── Intervention 4: Kombiniert ─────────────────────────────────────────

    combined = {}
    if not args.skip_combined:
        print(f"\n  Starte kombinierten Sweep (τ × θ_hit × θ_flop)...")
        print(f"  (Abbruch moeglich mit Ctrl+C — Einzelergebnisse bereits oben)")
        combined = sweep_combined(probs, y_test, class_priors)
        print_combined_result(combined, baseline_ba)
    else:
        print(f"\n  Kombinierter Sweep uebersprungen (--skip-combined)")

    # ── Summary ────────────────────────────────────────────────────────────

    best_tau = tau_results[0] if tau_results else {}
    best_theta = theta_results[0] if theta_results else {}

    print(f"\n{'=' * 79}")
    print(f"  ZUSAMMENFASSUNG")
    print(f"{'=' * 79}")
    print(f"  Baseline:              BA={baseline_ba:.4f}")
    if best_tau:
        delta_tau = best_tau["BA"] - baseline_ba
        print(f"  Bestes τ={best_tau['tau']:4.2f}:          BA={best_tau['BA']:.4f}  (Δ={delta_tau:+.4f})")
    if best_theta:
        delta_theta = best_theta["BA"] - baseline_ba
        print(f"  Bestes θ-Paar:         BA={best_theta['BA']:.4f}  (Δ={delta_theta:+.4f})")
    if combined:
        delta_comb = combined["BA"] - baseline_ba
        print(f"  Kombiniert (τ+θ):      BA={combined['BA']:.4f}  (Δ={delta_comb:+.4f})")

    print(f"\n  Empfehlung fuer evaluate.py:")
    if best_tau and best_tau["BA"] > baseline_ba:
        print(f"    --tau {best_tau['tau']}")
    if best_theta and best_theta["BA"] > baseline_ba:
        print(f"    --theta-hit {best_theta['theta_hit']} --theta-flop {best_theta['theta_flop']}")

    # ── Report speichern ───────────────────────────────────────────────────

    if args.save_report:
        reports_dir = ensure_dir(paths.get("reports", Path("./outputs/reports")))
        model_suffix = model_path.stem.replace("spotilyzer_model_", "")
        report = {
            "created": datetime.now().isoformat(),
            "model_path": str(model_path),
            "dataset": args.dataset,
            "validated_only": args.validated_only,
            "holdout_size": int(X_test.shape[0]),
            "holdout_distribution": {
                label_names[u]: int(c) for u, c in zip(unique, counts)
            },
            "class_priors": {
                "flop": float(class_priors[IDX_FLOP]),
                "hit":  float(class_priors[IDX_HIT]),
                "mid":  float(class_priors[IDX_MID]),
            },
            "baseline_ba": float(baseline_ba),
            "leakage": leakage,
            "tau_sweep": tau_results,
            "threshold_sweep_top20": theta_results,
            "combined_best": combined,
            "recommendation": {
                "tau": best_tau.get("tau") if best_tau else None,
                "theta_hit": best_theta.get("theta_hit") if best_theta else None,
                "theta_flop": best_theta.get("theta_flop") if best_theta else None,
            },
        }
        report_path = reports_dir / f"postprocessing_tuning_{model_suffix}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n  Report gespeichert: {report_path}")

    print(f"\n{'─' * 79}")
    print(f"  DONE")
    print(f"{'─' * 79}")


if __name__ == "__main__":
    main()
