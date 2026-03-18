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
from utils.metadata import read_tracks_as_dict, get_tracks_jsonl_path

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
    return bundle


def load_evaluation_data(
    embeddings_dir: Path,
    jsonl_path: Path,
    sample_weights_cfg: dict,
    validated_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Laedt Embeddings + Labels aus tracks.jsonl fuer Evaluation.

    Args:
        validated_only: Nur Tracks mit robustness='validated' evaluieren.

    Returns:
        (X, sample_weights, merged_df)
    """
    embeddings = np.load(embeddings_dir / "embeddings.npy")
    meta_df = pd.read_csv(embeddings_dir / "embeddings_meta.csv")
    tracks_dict = read_tracks_as_dict(jsonl_path)

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
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    model,
    label_encoder,
    X: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray = None,
) -> dict:
    """Fuehrt vollstaendige Evaluation durch."""
    y = label_encoder.transform(labels)
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)

    # Basis-Metriken
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "f1_macro": float(f1_score(y, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y, y_pred, average="weighted")),
    }

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

    # Prediction-Confidence-Statistiken
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

    metadata_dir = paths.get("metadata")
    jsonl_path = get_tracks_jsonl_path(metadata_dir)
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
        "--save-report",
        action="store_true",
        help="Speichere Report als JSON"
    )
    parser.add_argument(
        "--validated-only",
        action="store_true",
        help="Nur Tracks mit robustness='validated' evaluieren (contested + single_source werden ausgeschlossen)"
    )
    args = parser.parse_args()

    # --embedder überschreibt --model und --embeddings-dir
    if args.embedder:
        tag = _EMBEDDER_TAG[args.embedder]
        model_short = _EMBEDDER_HF[args.embedder].split("/")[-1]
        args.embeddings_dir = str(embeddings_base / model_short)
        # Neustes Modell mit passendem Embedder-Tag + optionalem validated-Filter
        if args.validated_only:
            tagged = sorted(models_dir.glob(f"spotilyzer_model_{tag}_validated_*.joblib"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        else:
            # Modelle ohne validated-Tag bevorzugen (nicht auf _validated_ matchen)
            all_tagged = sorted(models_dir.glob(f"spotilyzer_model_{tag}_*.joblib"),
                                key=lambda p: p.stat().st_mtime, reverse=True)
            tagged = [p for p in all_tagged if "_validated_" not in p.name]
            if not tagged:
                tagged = all_tagged  # Fallback: beliebiges Modell mit diesem Tag
        if tagged:
            args.model = str(tagged[0])
        # sonst bleibt default_model

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

    if not jsonl_path.exists():
        print(f"  Fehler: tracks.jsonl nicht gefunden: {jsonl_path}")
        logger.error(f"tracks.jsonl nicht gefunden: {jsonl_path}")
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

    # Daten laden
    print(f"\n  Lade Evaluations-Daten...")
    X, sample_weights, merged_df = load_evaluation_data(
        embeddings_dir, jsonl_path, sample_weights_cfg,
        validated_only=args.validated_only,
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
