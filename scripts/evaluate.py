"""
evaluate.py
===========
Standalone-Evaluation eines trainierten Modells.

Laedt ein gespeichertes .joblib-Modell und evaluiert es gegen
tracks.jsonl + Embeddings. Generiert detaillierte Metriken,
Confusion Matrix und vergleicht gegen Ziel-Metriken.

Input:
  - outputs/models/spotilyzer_model.joblib
  - outputs/embeddings/embeddings.npy + embeddings_meta.csv
  - metadata/tracks.jsonl (Labels + Robustheit)

Output:
  - outputs/reports/evaluation_report.json
  - Konsolenausgabe mit detaillierten Metriken
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
)
import joblib

from _utils import (
    setup_logging,
    load_paths_config,
    load_training_config,
    load_thresholds_config,
    ensure_dir,
)
from utils.metadata import read_tracks_as_dict, get_tracks_jsonl_path, read_tracks

# Logger (wird in main() initialisiert)
logger = None


# ══════════════════════════════════════════════════════════════════════════════
# DATEN LADEN
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_path: Path) -> dict:
    """Laedt das gespeicherte Modell-Bundle."""
    print(f"  Lade Modell: {model_path}")
    bundle = joblib.load(model_path)
    print(f"    Typ: {type(bundle['model']).__name__}")
    print(f"    Embedding-Dim: {bundle.get('embedding_dim', '?')}")
    n_test = len(bundle.get("test_track_ids", []))
    if n_test:
        print(f"    Test-Track-IDs: {n_test} gespeichert")
    else:
        print(f"    Test-Track-IDs: nicht im Bundle (altes Modell) — Evaluation auf allen Daten")
    return bundle


def load_tracks_from_datasets(jsonl_paths: list[Path]) -> dict[int, dict]:
    """Laedt und merged Tracks aus mehreren JSONL-Quellen."""
    merged = {}
    for p in jsonl_paths:
        if not p.exists():
            print(f"  WARNUNG: JSONL nicht gefunden, uebersprungen: {p}")
            continue
        for track in read_tracks(p):
            tid = track.get("track_id")
            if tid is not None:
                if tid in merged:
                    # Clusters mergen, Rest aktualisieren
                    old_clusters = set(merged[tid].get("clusters", []))
                    new_clusters = set(track.get("clusters", []))
                    merged[tid].update(track)
                    merged[tid]["clusters"] = sorted(old_clusters | new_clusters)
                else:
                    merged[tid] = track
    return merged


def load_evaluation_data(
    embeddings_dir: Path,
    jsonl_paths: list[Path],
    sample_weights_cfg: dict,
    validated_only: bool = False,
    test_track_ids: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Laedt Embeddings + Labels aus einer oder mehreren tracks.jsonl fuer Evaluation.

    Args:
        jsonl_paths:     Liste von JSONL-Pfaden (main + Dataset-Module).
        validated_only:  Nur Tracks mit robustness='validated' evaluieren.
        test_track_ids:  Wenn gesetzt, nur diese Track-IDs evaluieren (Holdout-Set).

    Returns:
        (X, sample_weights, merged_df)
    """
    embeddings = np.load(embeddings_dir / "embeddings.npy")
    meta_df = pd.read_csv(embeddings_dir / "embeddings_meta.csv")
    tracks_dict = load_tracks_from_datasets(jsonl_paths)

    test_ids_set = set(test_track_ids.tolist()) if test_track_ids is not None else None
    if test_ids_set is not None:
        print(f"  Filter: nur Holdout-Testset ({len(test_ids_set)} Track-IDs aus Modell-Bundle)")
    if validated_only:
        print("  Filter: nur validated-Tracks (contested + single_source werden uebersprungen)")

    labels = []
    robustness_list = []
    weights = []
    valid_mask = []
    skipped_robustness = 0

    for _, row in meta_df.iterrows():
        tid = int(row["track_id"])
        track = tracks_dict.get(tid, {})
        label = track.get("label")
        robustness = track.get("robustness", "single_source")

        if label is None:
            valid_mask.append(False)
            continue

        if test_ids_set is not None and tid not in test_ids_set:
            valid_mask.append(False)
            continue

        if validated_only and robustness != "validated":
            skipped_robustness += 1
            valid_mask.append(False)
            continue

        labels.append(label)
        robustness_list.append(robustness)
        weights.append(sample_weights_cfg.get(robustness, 0.5))
        valid_mask.append(True)

    if skipped_robustness > 0:
        print(f"    {skipped_robustness} Tracks durch validated_only-Filter entfernt")

    valid_mask = np.array(valid_mask)
    valid_indices = meta_df.loc[valid_mask, "embedding_idx"].values
    X = embeddings[valid_indices]
    sample_weights = np.array(weights, dtype=np.float32)

    merged = meta_df[valid_mask].copy()
    merged["label"] = labels
    merged["robustness"] = robustness_list
    merged["sample_weight"] = sample_weights

    return X, sample_weights, merged


# ══════════════════════════════════════════════════════════════════════════════
# POST-HOC ADJUSTMENTS
# Klassen-Index: flop=0, hit=1, mid=2 (LabelEncoder alphabetisch)
# predict_proba Reihenfolge: [p_flop, p_hit, p_mid]
# ══════════════════════════════════════════════════════════════════════════════

def _logit_adjustment(probs: np.ndarray, class_priors: np.ndarray, tau: float) -> np.ndarray:
    """p'(c) ∝ p(c) / π(c)^τ  — re-normalisiert auf Summe=1."""
    if tau == 0.0:
        return probs.copy()
    adjustment = np.clip(class_priors, 1e-9, None) ** tau
    adjusted = np.clip(probs / adjustment, 1e-12, None)
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def _two_threshold_predict(probs: np.ndarray, theta_hit: float, theta_flop: float) -> np.ndarray:
    """Mid als Default; Hit wenn p(hit)>θ_hit; Flop wenn p(flop)>θ_flop (und nicht Hit)."""
    preds = np.full(len(probs), 2, dtype=int)   # Default: mid=2
    preds[probs[:, 1] > theta_hit] = 1           # hit=1
    flop_mask = (probs[:, 0] > theta_flop) & (preds != 1)
    preds[flop_mask] = 0                          # flop=0
    return preds


def _compute_class_priors_from_labels(y: np.ndarray) -> np.ndarray:
    """Berechnet Priors aus int-Labels [0=flop,1=hit,2=mid]."""
    priors = np.array([
        (y == 0).mean(),
        (y == 1).mean(),
        (y == 2).mean(),
    ])
    return priors


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    model,
    label_encoder,
    X: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray = None,
    tau: float = None,
    theta_hit: float = None,
    theta_flop: float = None,
) -> dict:
    """
    Fuehrt vollstaendige Evaluation durch.

    Optionale Post-hoc Adjustments (aus tune_postprocessing.py ermitteln):
        tau:        Logit-Adjustment Staerke (0=off; typ. 0.5-1.5)
        theta_hit:  Schwelle fuer Hit-Entscheidung (None=argmax)
        theta_flop: Schwelle fuer Flop-Entscheidung (None=argmax)
    """
    y = label_encoder.transform(labels)
    y_pred_proba = model.predict_proba(X)

    # Post-hoc Adjustments anwenden
    probs = y_pred_proba.copy()
    postprocess_info = {}

    if tau is not None and tau > 0.0:
        class_priors = _compute_class_priors_from_labels(y)
        probs = _logit_adjustment(probs, class_priors, tau)
        postprocess_info["tau"] = tau
        postprocess_info["class_priors"] = {
            "flop": float(class_priors[0]),
            "hit":  float(class_priors[1]),
            "mid":  float(class_priors[2]),
        }

    if theta_hit is not None and theta_flop is not None:
        y_pred = _two_threshold_predict(probs, theta_hit, theta_flop)
        postprocess_info["theta_hit"] = theta_hit
        postprocess_info["theta_flop"] = theta_flop
    else:
        y_pred = np.argmax(probs, axis=1)

    if postprocess_info:
        print(f"  Post-hoc Adjustment: {postprocess_info}")

    # Basis-Metriken
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "f1_macro": float(f1_score(y, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y, y_pred, average="weighted")),
    }

    if postprocess_info:
        metrics["postprocess"] = postprocess_info

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    # Per-Class Metrics
    # LabelEncoder sortiert alphabetisch: flop=0, hit=1, mid=2
    report = classification_report(
        y, y_pred,
        target_names=["flop", "hit", "mid"],
        output_dict=True,
    )
    metrics["per_class"] = {}
    for cls in ["flop", "hit", "mid"]:
        metrics["per_class"][cls] = {
            "precision": float(report[cls]["precision"]),
            "recall": float(report[cls]["recall"]),
            "f1": float(report[cls]["f1-score"]),
            "support": int(report[cls]["support"]),
        }

    # Prediction-Confidence-Statistiken (aus Original-Probs, vor Adjustment)
    max_proba = y_pred_proba.max(axis=1)
    metrics["confidence_stats"] = {
        "mean": float(max_proba.mean()),
        "median": float(np.median(max_proba)),
        "min": float(max_proba.min()),
        "max": float(max_proba.max()),
    }

    # Per-Robustness Evaluation
    if sample_weights is not None:
        robustness_metrics = {}
        for weight_val, rob_name in [(1.0, "validated"), (0.5, "single_source"), (0.7, "contested")]:
            mask = np.isclose(sample_weights, weight_val, atol=0.05)
            if mask.sum() > 0:
                rob_acc = float(balanced_accuracy_score(y[mask], y_pred[mask]))
                robustness_metrics[rob_name] = {
                    "balanced_accuracy": rob_acc,
                    "count": int(mask.sum()),
                }
        metrics["per_robustness"] = robustness_metrics

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def print_evaluation_report(metrics: dict, target_metrics: dict = None):
    """Gibt detaillierten Evaluations-Report aus."""

    print(f"\n{'=' * 79}")
    print(f"  EVALUATION REPORT")
    print(f"{'=' * 79}")

    print(f"\n  Overall Metrics:")
    print(f"    Accuracy:          {metrics['accuracy']:.3f}")
    print(f"    Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
    print(f"    F1 (macro):        {metrics['f1_macro']:.3f}")
    print(f"    F1 (weighted):     {metrics['f1_weighted']:.3f}")

    print(f"\n  Per-Class Performance:")
    print(f"    {'Class':8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"    {'-' * 48}")
    for cls in ["flop", "hit", "mid"]:
        m = metrics["per_class"][cls]
        print(f"    {cls:8} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f} {m['support']:>10}")

    print(f"\n  Confusion Matrix:")
    print(f"    {'':8} {'flop':>8} {'hit':>8} {'mid':>8}  <- predicted")
    cm = metrics["confusion_matrix"]
    for i, cls in enumerate(["flop", "hit", "mid"]):
        row = f"    {cls:8}"
        for j in range(3):
            row += f" {cm[i][j]:>7}"
        print(row)
    print(f"    ^ actual")

    # Confidence-Statistiken
    if "confidence_stats" in metrics:
        cs = metrics["confidence_stats"]
        print(f"\n  Prediction Confidence:")
        print(f"    Mean:   {cs['mean']:.3f}")
        print(f"    Median: {cs['median']:.3f}")
        print(f"    Min:    {cs['min']:.3f}")
        print(f"    Max:    {cs['max']:.3f}")

    # Per-Robustness
    if "per_robustness" in metrics:
        print(f"\n  Per-Robustness Balanced Accuracy:")
        for rob, data in metrics["per_robustness"].items():
            print(f"    {rob:15}: {data['balanced_accuracy']:.3f}  (n={data['count']})")

    # Target-Vergleich
    if target_metrics:
        print(f"\n{'─' * 79}")
        print(f"  ZIEL-VERGLEICH")
        print(f"{'─' * 79}")

        flop_recall = metrics["per_class"]["flop"]["recall"]
        hit_recall = metrics["per_class"]["hit"]["recall"]
        ba = metrics["balanced_accuracy"]

        flop_target = target_metrics.get("flop_recall_min", 0.5)
        hit_target = target_metrics.get("hit_recall_min", 0.8)
        ba_target = target_metrics.get("balanced_accuracy_min", 0.65)

        def status(val, target):
            return "OK" if val >= target else "MISS"

        print(f"    {'Metrik':25} {'Aktuell':>10} {'Ziel':>10} {'Status':>10}")
        print(f"    {'-' * 55}")
        print(f"    {'Flop Recall':25} {flop_recall:>10.3f} {'>=' + str(flop_target):>10} {status(flop_recall, flop_target):>10}")
        print(f"    {'Hit Recall':25} {hit_recall:>10.3f} {'>=' + str(hit_target):>10} {status(hit_recall, hit_target):>10}")
        print(f"    {'Balanced Accuracy':25} {ba:>10.3f} {'>=' + str(ba_target):>10} {status(ba, ba_target):>10}")

        all_ok = (flop_recall >= flop_target and hit_recall >= hit_target and ba >= ba_target)
        print(f"\n    Gesamt: {'ALLE ZIELE ERREICHT' if all_ok else 'ZIELE NOCH NICHT ERREICHT'}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global logger

    paths = load_paths_config()
    training_cfg = load_training_config()
    thresholds_cfg = load_thresholds_config()

    models_dir = paths.get("models", Path("./outputs/models"))
    embeddings_base = paths.get("embeddings", Path("./outputs/embeddings"))

    _EMBEDDER_HF  = {"95M": "m-a-p/MERT-v1-95M", "330M": "m-a-p/MERT-v1-330M"}
    _EMBEDDER_TAG = {"95M": "MERTv195M",           "330M": "MERTv1330M"}

    # Neuestes Modell als Default (ohne Embedder-Filter)
    _glob_all = sorted(models_dir.glob("spotilyzer_model_*.joblib"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    default_model = str(_glob_all[0] if _glob_all else models_dir / "spotilyzer_model.joblib")

    # Default embeddings: aus training.yaml ableiten
    cfg_model = training_cfg.get("embedder", {}).get("model", "")
    cfg_short = cfg_model.split("/")[-1] if cfg_model else ""
    default_embeddings = str(embeddings_base / cfg_short if cfg_short else embeddings_base)

    parser = argparse.ArgumentParser(
        description="Evaluiere trainiertes Spotilyzer-Modell"
    )
    parser.add_argument(
        "--embedder",
        choices=list(_EMBEDDER_HF.keys()),
        default=None,
        help="Embedder-Kurzname (95M | 330M) — leitet --embeddings-dir und --model automatisch ab"
    )
    parser.add_argument(
        "--model",
        default=default_model,
        help=f"Pfad zum .joblib-Modell (default: neuestes in models/)"
    )
    parser.add_argument(
        "--embeddings-dir",
        default=default_embeddings,
        help=f"Verzeichnis mit Embeddings (default: aus training.yaml = {default_embeddings})"
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["main"],
        metavar="DATASET",
        help="Datensatz-Quellen: 'main' und/oder Modul-Namen z.B. 'spotify_charts' (default: main)"
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Speichere Report als JSON"
    )
    parser.add_argument(
        "--validated-only",
        action="store_true",
        help="Nur Tracks mit robustness='validated' evaluieren (contested + single_source werden ausgeschlossen)"
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        metavar="TAU",
        help="Logit-Adjustment Staerke (0.0=off; optimal via tune_postprocessing.py ermitteln)"
    )
    parser.add_argument(
        "--theta-hit",
        type=float,
        default=None,
        metavar="THETA",
        help="Zweistufen-Schwelle fuer Hit (zusammen mit --theta-flop; optimal via tune_postprocessing.py)"
    )
    parser.add_argument(
        "--theta-flop",
        type=float,
        default=None,
        metavar="THETA",
        help="Zweistufen-Schwelle fuer Flop (zusammen mit --theta-hit)"
    )
    args = parser.parse_args()

    # Plausibilitaets-Check: Schwellen muessen gemeinsam gesetzt sein
    if (args.theta_hit is None) != (args.theta_flop is None):
        print("  Fehler: --theta-hit und --theta-flop muessen gemeinsam gesetzt werden.")
        sys.exit(1)

    # --embedder setzt embeddings-dir und (wenn kein explizites --model) das Modell automatisch
    if args.embedder:
        tag = _EMBEDDER_TAG[args.embedder]
        model_short = _EMBEDDER_HF[args.embedder].split("/")[-1]
        args.embeddings_dir = str(embeddings_base / model_short)
        # Modell-Autodetect nur wenn kein explizites --model übergeben wurde
        if args.model == default_model:
            if args.validated_only:
                # *validated* matcht auch "main+spotify_charts_validated" etc.
                tagged = sorted(models_dir.glob(f"spotilyzer_model_{tag}*validated*.joblib"),
                                key=lambda p: p.stat().st_mtime, reverse=True)
            else:
                all_tagged = sorted(models_dir.glob(f"spotilyzer_model_{tag}*.joblib"),
                                    key=lambda p: p.stat().st_mtime, reverse=True)
                tagged = [p for p in all_tagged if "validated" not in p.name]
                if not tagged:
                    tagged = all_tagged
            if tagged:
                args.model = str(tagged[0])

    # JSONL-Pfade aus --dataset aufbauen
    datasets_base = paths.get("datasets", Path("G:/Dev/SpotilyzerData/datasets"))
    metadata_dir = paths.get("metadata")
    jsonl_paths = []
    for ds in args.dataset:
        if ds == "main":
            jsonl_paths.append(get_tracks_jsonl_path(metadata_dir))
        else:
            jsonl_paths.append(Path(datasets_base) / ds / "tracks.jsonl")

    # Logging
    logger = setup_logging("evaluation", log_dir=paths.get("logs"))

    print(f"{'=' * 79}")
    print(f"  SPOTILYZER MODEL EVALUATION")
    print(f"{'=' * 79}")
    if args.embedder:
        print(f"  Embedder:  {args.embedder}  ({_EMBEDDER_HF[args.embedder]})")
    if args.validated_only:
        print(f"  Filter:    validated_only=True  (contested + single_source werden ausgeschlossen)")

    model_path = Path(args.model)
    embeddings_dir = Path(args.embeddings_dir)

    # Pruefen
    if not model_path.exists():
        print(f"  Fehler: Modell nicht gefunden: {model_path}")
        logger.error(f"Modell nicht gefunden: {model_path}")
        sys.exit(1)

    if not (embeddings_dir / "embeddings.npy").exists():
        print(f"  Fehler: Embeddings nicht gefunden: {embeddings_dir}")
        logger.error(f"Embeddings nicht gefunden: {embeddings_dir}")
        sys.exit(1)

    # Sample-Weights aus thresholds.yaml
    sample_weights_cfg = thresholds_cfg.get("sample_weights", {
        "validated": 1.0,
        "single_source": 0.5,
        "contested": 0.7,
    })

    # Modell laden
    bundle = load_model(model_path)
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]
    test_track_ids = bundle.get("test_track_ids", None)

    # Daten laden
    print(f"\n  Lade Evaluations-Daten...")
    print(f"  Datasets: {args.dataset}")
    X, sample_weights, merged_df = load_evaluation_data(
        embeddings_dir, jsonl_paths, sample_weights_cfg,
        validated_only=args.validated_only,
        test_track_ids=test_track_ids,
    )
    print(f"    Samples: {X.shape[0]}")
    print(f"    Features: {X.shape[1]}")

    # Evaluation
    print(f"\n{'─' * 79}")
    print(f"  EVALUATION")
    print(f"{'─' * 79}")

    metrics = evaluate_model(
        model, label_encoder,
        X, merged_df["label"].values,
        sample_weights,
        tau=args.tau,
        theta_hit=args.theta_hit,
        theta_flop=args.theta_flop,
    )

    target_metrics = training_cfg.get("target_metrics", {})
    print_evaluation_report(metrics, target_metrics)

    # Report speichern
    if args.save_report:
        reports_dir = ensure_dir(paths.get("reports", Path("./outputs/reports")))
        report = {
            "created": datetime.now().isoformat(),
            "model_path": str(model_path),
            "dataset": {
                "n_samples": int(X.shape[0]),
                "label_distribution": {k: int(v) for k, v in merged_df["label"].value_counts().items()},
                "robustness_distribution": {k: int(v) for k, v in merged_df["robustness"].value_counts().items()},
            },
            "metrics": metrics,
            "target_metrics": target_metrics,
        }

        model_suffix = model_path.stem.replace("spotilyzer_model_", "")
        report_path = reports_dir / f"evaluation_report_{model_suffix}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n  Report gespeichert: {report_path}")
        logger.info(f"Evaluation Report gespeichert: {report_path}")

    logger.info(
        f"Evaluation abgeschlossen: BA={metrics['balanced_accuracy']:.3f}, "
        f"Flop-Recall={metrics['per_class']['flop']['recall']:.3f}"
    )

    print(f"\n{'─' * 79}")
    print(f"  DONE")
    print(f"{'─' * 79}")


if __name__ == "__main__":
    main()
