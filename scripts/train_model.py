"""
train_model.py
==============
Trainiert einen XGBoost-Klassifikator auf MERT-Embeddings.

Verwendet tracks.jsonl mit Multi-Source-Labels und sample_weight
aus Robustheit (validated/single_source/contested).

Input:
  - outputs/embeddings/embeddings.npy (MERT-Embeddings)
  - outputs/embeddings/embeddings_meta.csv (Track-IDs)
  - metadata/tracks.jsonl (Labels + Robustheit)

Output:
  - outputs/models/spotilyzer_model.joblib (trainiertes Modell)
  - outputs/reports/training_report.json (Metriken, Confusion Matrix)
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
)
import xgboost as xgb
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
# DATEN-VORBEREITUNG
# ══════════════════════════════════════════════════════════════════════════════

def load_data(
    embeddings_dir: Path,
    jsonl_path: Path,
    sample_weights_cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, LabelEncoder]:
    """
    Laedt Embeddings und merged sie mit Labels aus tracks.jsonl.

    Returns:
        (X, y, sample_weights, metadata_df, label_encoder)
    """
    print("  Lade Embeddings...")
    embeddings = np.load(embeddings_dir / "embeddings.npy")
    meta_df = pd.read_csv(embeddings_dir / "embeddings_meta.csv")

    print(f"    Embeddings: {embeddings.shape}")
    print(f"    Metadaten:  {len(meta_df)} Eintraege")

    print("  Lade Labels aus tracks.jsonl...")
    tracks_dict = read_tracks_as_dict(jsonl_path)

    # Label und Robustheit aus JSONL holen
    labels = []
    robustness_list = []
    valid_mask = []

    for _, row in meta_df.iterrows():
        tid = int(row["track_id"])
        track = tracks_dict.get(tid, {})
        label = track.get("label")
        robustness = track.get("robustness", "single_source")

        if label is not None:
            labels.append(label)
            robustness_list.append(robustness)
            valid_mask.append(True)
        else:
            valid_mask.append(False)

    valid_mask = np.array(valid_mask)
    dropped = (~valid_mask).sum()
    if dropped > 0:
        print(f"    {dropped} Tracks ohne Label entfernt")
        if logger:
            logger.warning(f"{dropped} Tracks ohne Label entfernt")

    # Filtern
    valid_indices = meta_df.loc[valid_mask, "embedding_idx"].values
    X = embeddings[valid_indices]

    # Sample Weights aus Robustheit berechnen
    sample_weights = np.array([
        sample_weights_cfg.get(rob, 0.5)
        for rob in robustness_list
    ], dtype=np.float32)

    # Label-Encoding
    label_encoder = LabelEncoder()
    label_encoder.fit(["flop", "mid", "hit"])  # Feste Reihenfolge
    y = label_encoder.transform(labels)

    # Metadata-DF fuer Reporting
    meta_filtered = meta_df[valid_mask].copy()
    meta_filtered["label"] = labels
    meta_filtered["robustness"] = robustness_list
    meta_filtered["sample_weight"] = sample_weights

    print(f"\n  Label-Verteilung:")
    for label in ["flop", "mid", "hit"]:
        count = sum(1 for l in labels if l == label)
        pct = 100 * count / len(labels)
        print(f"    {label:5}: {count:>5} ({pct:.1f}%)")

    print(f"\n  Robustheit:")
    for rob in ["validated", "single_source", "contested"]:
        count = sum(1 for r in robustness_list if r == rob)
        pct = 100 * count / len(robustness_list)
        print(f"    {rob:15}: {count:>5} ({pct:.1f}%)")

    if logger:
        logger.info(
            f"Daten geladen: {X.shape[0]} Samples, {X.shape[1]} Features"
        )

    return X, y, sample_weights, meta_filtered, label_encoder


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray,
    xgb_params: dict,
    use_gpu: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
    early_stopping_rounds: int = 30,
) -> tuple[xgb.XGBClassifier, dict]:
    """
    Trainiert XGBoost-Modell mit sample_weight.

    Returns:
        (model, metrics_dict)
    """
    print(f"\n  Train/Test Split ({int((1-test_size)*100)}/{int(test_size*100)})...")
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, sample_weights,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"    Train: {len(X_train)}, Test: {len(X_test)}")

    # Model konfigurieren
    params = xgb_params.copy()
    params["random_state"] = random_state
    params["n_jobs"] = -1
    params["tree_method"] = "hist"

    if use_gpu:
        params["device"] = "cuda"
        print("  Training auf GPU...")
    else:
        print("  Training auf CPU...")

    if logger:
        logger.info(f"Training gestartet: {len(X_train)} Train, {len(X_test)} Test")

    model = xgb.XGBClassifier(**params, early_stopping_rounds=early_stopping_rounds)

    # Training MIT sample_weight (Robustheit × Klassenbalancierung)
    class_weights = compute_sample_weight("balanced", y_train)
    combined_weights = w_train * class_weights
    print(f"  Trainiere XGBoost (sample_weight balanced, early_stopping={early_stopping_rounds})...")
    model.fit(
        X_train, y_train,
        sample_weight=combined_weights,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )
    best = getattr(model, "best_iteration", None)
    if best is not None:
        print(f"  Best iteration: {best + 1} / {xgb_params.get('n_estimators', '?')}")

    # Predictions
    y_pred = model.predict(X_test)

    # Metriken
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
    }

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    # Per-Class Metrics
    # LabelEncoder sortiert alphabetisch: flop=0, hit=1, mid=2
    report = classification_report(
        y_test, y_pred,
        target_names=["flop", "hit", "mid"],
        output_dict=True,
    )
    metrics["per_class"] = {
        cls: {
            "precision": float(report[cls]["precision"]),
            "recall": float(report[cls]["recall"]),
            "f1": float(report[cls]["f1-score"]),
            "support": int(report[cls]["support"]),
        }
        for cls in ["flop", "hit", "mid"]
    }

    # Cross-Validation mit sample_weight
    print("  Cross-Validation (5-fold) mit sample_weight...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        fold_model = xgb.XGBClassifier(**params, early_stopping_rounds=early_stopping_rounds)
        fold_class_weights = compute_sample_weight("balanced", y[train_idx])
        fold_combined = sample_weights[train_idx] * fold_class_weights
        fold_model.fit(
            X[train_idx], y[train_idx],
            sample_weight=fold_combined,
            eval_set=[(X[val_idx], y[val_idx])],
            verbose=False,
        )
        fold_pred = fold_model.predict(X[val_idx])
        fold_score = balanced_accuracy_score(y[val_idx], fold_pred)
        cv_scores.append(fold_score)

    cv_scores = np.array(cv_scores)
    metrics["cv_balanced_accuracy_mean"] = float(cv_scores.mean())
    metrics["cv_balanced_accuracy_std"] = float(cv_scores.std())

    if logger:
        logger.info(
            f"Training abgeschlossen: BA={metrics['balanced_accuracy']:.3f}, "
            f"CV={cv_scores.mean():.3f}+/-{cv_scores.std():.3f}"
        )

    return model, metrics


# ══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def print_report(metrics: dict, target_metrics: dict = None):
    """Gibt einen formatierten Trainingsreport aus."""

    print(f"\n{'=' * 79}")
    print(f"  TRAINING REPORT")
    print(f"{'=' * 79}")

    print(f"\n  Overall Metrics:")
    print(f"    Accuracy:          {metrics['accuracy']:.3f}")
    print(f"    Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
    print(f"    F1 (macro):        {metrics['f1_macro']:.3f}")
    print(f"    F1 (weighted):     {metrics['f1_weighted']:.3f}")

    print(f"\n  Cross-Validation (5-fold):")
    print(f"    Balanced Accuracy: {metrics['cv_balanced_accuracy_mean']:.3f} +/- {metrics['cv_balanced_accuracy_std']:.3f}")

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

    # Target-Vergleich
    if target_metrics:
        print(f"\n  Ziel-Vergleich:")
        flop_recall = metrics["per_class"]["flop"]["recall"]
        hit_recall = metrics["per_class"]["hit"]["recall"]
        ba = metrics["balanced_accuracy"]

        flop_target = target_metrics.get("flop_recall_min", 0.5)
        hit_target = target_metrics.get("hit_recall_min", 0.8)
        ba_target = target_metrics.get("balanced_accuracy_min", 0.65)

        def check(val, target):
            return "OK" if val >= target else "MISS"

        print(f"    Flop Recall:       {flop_recall:.3f}  (Ziel: >={flop_target})  [{check(flop_recall, flop_target)}]")
        print(f"    Hit Recall:        {hit_recall:.3f}  (Ziel: >={hit_target})  [{check(hit_recall, hit_target)}]")
        print(f"    Balanced Accuracy: {ba:.3f}  (Ziel: >={ba_target})  [{check(ba, ba_target)}]")


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
    embeddings_base = paths.get("embeddings", Path("./outputs/embeddings"))
    # Embedder-Modell aus training.yaml → Subverzeichnis ableiten
    cfg_model = training_cfg.get("embedder", {}).get("model", "")
    model_short = cfg_model.split("/")[-1] if cfg_model else ""
    if model_short:
        default_embeddings = str(Path(embeddings_base) / model_short)
    else:
        default_embeddings = str(embeddings_base)
    default_output = str(paths.get("models", "./outputs/models"))

    _EMBEDDER_HF = {"95M": "m-a-p/MERT-v1-95M", "330M": "m-a-p/MERT-v1-330M"}

    parser = argparse.ArgumentParser(
        description="Trainiere XGBoost-Klassifikator auf MERT-Embeddings (mit sample_weight)"
    )
    parser.add_argument(
        "--embedder",
        choices=list(_EMBEDDER_HF.keys()),
        default=None,
        help="Embedder-Kurzname (95M | 330M) — leitet --embeddings-dir automatisch ab"
    )
    parser.add_argument(
        "--embeddings-dir",
        default=default_embeddings,
        help=f"Verzeichnis mit Embeddings (default: aus training.yaml = {default_embeddings})"
    )
    parser.add_argument(
        "--output-dir",
        default=default_output,
        help=f"Output-Verzeichnis fuer Modell (default: {default_output})"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="GPU fuer XGBoost verwenden"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test-Set Groesse (default: 0.2)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Nur Daten laden, nicht trainieren"
    )
    args = parser.parse_args()

    # --embedder überschreibt --embeddings-dir (Kurzform hat Priorität)
    if args.embedder:
        hf_name = _EMBEDDER_HF[args.embedder]
        model_short = hf_name.split("/")[-1]
        args.embeddings_dir = str(Path(embeddings_base) / model_short)
        cfg_model = hf_name  # für Anzeige

    # Logging
    logger = setup_logging("training", log_dir=paths.get("logs"))

    print(f"{'=' * 79}")
    print(f"  SPOTILYZER MODEL TRAINING (mit sample_weight)")
    print(f"{'=' * 79}")
    src = "CLI --embedder" if args.embedder else "training.yaml"
    if cfg_model:
        print(f"  Embedder:  {cfg_model}  ({src})")

    embeddings_dir = Path(args.embeddings_dir)
    output_dir = Path(args.output_dir)

    # Pruefen ob Dateien existieren
    if not (embeddings_dir / "embeddings.npy").exists():
        print(f"  Fehler: Embeddings nicht gefunden: {embeddings_dir / 'embeddings.npy'}")
        print(f"  -> Erst 'python scripts/extract_embeddings.py' ausfuehren!")
        logger.error(f"Embeddings nicht gefunden: {embeddings_dir}")
        sys.exit(1)

    if not jsonl_path.exists():
        print(f"  Fehler: tracks.jsonl nicht gefunden: {jsonl_path}")
        print(f"  -> Erst Pipeline ausfuehren (scout + enrich + labels)!")
        logger.error(f"tracks.jsonl nicht gefunden: {jsonl_path}")
        sys.exit(1)

    # Sample-Weights aus thresholds.yaml
    sample_weights_cfg = thresholds_cfg.get("sample_weights", {
        "validated": 1.0,
        "single_source": 0.5,
        "contested": 0.7,
    })

    # Daten laden
    print(f"\n{'─' * 79}")
    print(f"  DATEN LADEN")
    print(f"{'─' * 79}")

    X, y, sample_weights, meta_df, label_encoder = load_data(
        embeddings_dir, jsonl_path, sample_weights_cfg,
    )

    print(f"\n  Dataset:")
    print(f"    Features:       {X.shape[1]} (MERT embedding dim)")
    print(f"    Samples:        {X.shape[0]}")
    print(f"    Classes:        {len(label_encoder.classes_)} ({', '.join(label_encoder.classes_)})")
    print(f"    sample_weight:  min={sample_weights.min():.2f}, max={sample_weights.max():.2f}, mean={sample_weights.mean():.2f}")

    if args.dry_run:
        print(f"\n  [DRY RUN] Training uebersprungen.")
        return

    # XGBoost-Parameter aus Config
    model_cfg = training_cfg.get("model", {})
    xgb_params = model_cfg.get("params", {})
    random_state = training_cfg.get("random_state", 42)
    target_metrics = training_cfg.get("target_metrics", {})
    early_stopping_rounds = training_cfg.get("early_stopping_rounds", 30)

    # Training
    print(f"\n{'─' * 79}")
    print(f"  TRAINING")
    print(f"{'─' * 79}")

    model, metrics = train_model(
        X, y, sample_weights,
        xgb_params=xgb_params,
        use_gpu=args.gpu,
        test_size=args.test_size,
        random_state=random_state,
        early_stopping_rounds=early_stopping_rounds,
    )

    # Report
    print_report(metrics, target_metrics)

    # Modell speichern
    ensure_dir(output_dir)
    reports_dir = ensure_dir(paths.get("reports", output_dir.parent / "reports"))

    # Embedder-Kurznamen aus embeddings_info.json ableiten
    embedder_tag = f"{X.shape[1]}dim"  # Fallback: Feature-Dimension
    embeddings_info_path = embeddings_dir / "embeddings_info.json"
    if embeddings_info_path.exists():
        try:
            with open(embeddings_info_path, "r", encoding="utf-8") as f:
                emb_info = json.load(f)
            raw = emb_info.get("model", "")  # z.B. "m-a-p/MERT-v1-330M"
            short = raw.split("/")[-1]        # "MERT-v1-330M"
            embedder_tag = short.replace("-", "").replace(".", "")  # "MERTv1330M"
        except Exception:
            pass

    date_tag = datetime.now().strftime("%Y%m%d")
    model_filename = f"spotilyzer_model_{embedder_tag}_{date_tag}.joblib"
    model_path = output_dir / model_filename

    model_data = {
        "model": model,
        "label_encoder": label_encoder,
        "embedding_dim": X.shape[1],
        "training_config": training_cfg,
        "thresholds": thresholds_cfg,
    }
    joblib.dump(model_data, model_path)
    print(f"\n  Modell gespeichert: {model_path}")

    # Report speichern
    report = {
        "created": datetime.now().isoformat(),
        "dataset": {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "label_distribution": {k: int(v) for k, v in meta_df["label"].value_counts().items()},
            "robustness_distribution": {k: int(v) for k, v in meta_df["robustness"].value_counts().items()},
            "sample_weight_stats": {
                "min": float(sample_weights.min()),
                "max": float(sample_weights.max()),
                "mean": float(sample_weights.mean()),
            },
        },
        "model": {
            "type": "XGBClassifier",
            "params": xgb_params,
            "sample_weight": True,
        },
        "metrics": metrics,
        "target_metrics": target_metrics,
    }

    report_path = reports_dir / "training_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Report gespeichert: {report_path}")

    logger.info(f"Modell gespeichert: {model_path}")
    logger.info(f"Report gespeichert: {report_path}")

    print(f"\n{'─' * 79}")
    print(f"  DONE")
    print(f"{'─' * 79}")


if __name__ == "__main__":
    main()
