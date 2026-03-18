"""
run_pipeline.py
===============
Orchestrierungs-Skript fuer die SpotilyzerTraining-Pipeline.

Zwei Modi:
1. Interaktives Menue (Default, wenn ohne Argumente aufgerufen)
2. CLI-Flags fuer Automation (--full, --scout, --enrich, --train, etc.)

Datenfluss (JSONL-basiert):
  1. scout_deezer.py      → metadata/tracks.jsonl (track_id, title, artist, clusters, deezer_rank)
  2. download_previews.py → previews/{shard}/{track_id}.mp3 + tracks.jsonl (file_path)
  3. enrich_lastfm.py     → tracks.jsonl (lastfm_* Felder)
  4. compute_labels.py    → tracks.jsonl (label, robustness)
  5. extract_embeddings.py→ outputs/embeddings/embeddings.npy + embeddings_meta.csv
  6. train_model.py       → outputs/models/spotilyzer_model.joblib
  7. evaluate.py          → outputs/reports/evaluation_report.json
"""

import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

from _utils import (
    setup_logging,
    load_paths_config,
    load_training_config,
    ensure_dir,
)
from utils.metadata import get_tracks_jsonl_path

# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

SCRIPTS_DIR = Path(__file__).resolve().parent

# Verfuegbare Embedder-Modelle (Kurzname → vollstaendiger HF-Name)
SUPPORTED_MODELS = {
    "95M":  "m-a-p/MERT-v1-95M",
    "330M": "m-a-p/MERT-v1-330M",
}

PIPELINE_STEPS = {
    "scout": {
        "script": "scout_deezer.py",
        "name": "Deezer-Scouting",
        "description": "Genre-Cluster + Charts scouten, Tracks in JSONL mergen",
    },
    "download": {
        "script": "download_previews.py",
        "name": "Preview-Download",
        "description": "30s-MP3-Previews mit MD5-Sharding + ID3-Tags herunterladen",
    },
    "enrich": {
        "script": "enrich_lastfm.py",
        "name": "Last.fm-Enrichment",
        "description": "Tracks mit Last.fm playcount/listeners anreichern",
    },
    "labels": {
        "script": "compute_labels.py",
        "name": "Label-Berechnung",
        "description": "Multi-Source-Labels + Robustheit berechnen",
    },
    "embeddings": {
        "script": "extract_embeddings.py",
        "name": "MERT-Embedding-Extraktion",
        "description": "Audio-Embeddings mit konfiguriertem MERT-Modell extrahieren (siehe training.yaml)",
    },
    "train": {
        "script": "train_model.py",
        "name": "XGBoost-Training",
        "description": "Modell mit sample_weight trainieren",
    },
    "evaluate": {
        "script": "evaluate.py",
        "name": "Evaluation",
        "description": "Modell evaluieren, Metriken pruefen",
    },
}

# Reihenfolge der Pipeline
PIPELINE_ORDER = ["scout", "download", "enrich", "labels", "embeddings", "train", "evaluate"]

# Logger
logger = None


def _default_model_key() -> str:
    """Liest den konfigurierten Embedder aus training.yaml und gibt den Kurznamen zurueck."""
    try:
        cfg = load_training_config()
        hf_name = cfg.get("embedder", {}).get("model", "")
        return next((k for k, v in SUPPORTED_MODELS.items() if v == hf_name), "95M")
    except Exception:
        return "95M"


def build_model_step_args(model_key: str, base_step_args: dict, validated_only: bool = False) -> dict:
    """
    Fuegt modellspezifische CLI-Argumente zu den Schritt-Argumenten hinzu.

    Schritte die betroffen sind:
      embeddings → --model <hf-name>
      train      → --embedder <key>  [--validated-only]
      evaluate   → --embedder <key>  [--validated-only]
    """
    step_args = {k: list(v) for k, v in base_step_args.items()}
    step_args["embeddings"] = ["--model", SUPPORTED_MODELS[model_key]]
    step_args["train"]      = step_args.get("train", []) + ["--embedder", model_key]
    step_args["evaluate"]   = step_args.get("evaluate", ["--save-report"]) + ["--embedder", model_key]
    if validated_only:
        step_args["train"]    = step_args["train"] + ["--validated-only"]
        step_args["evaluate"] = step_args["evaluate"] + ["--validated-only"]
    return step_args


# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT-AUSFUEHRUNG
# ══════════════════════════════════════════════════════════════════════════════

def run_step(step_id: str, extra_args: list[str] = None) -> bool:
    """
    Fuehrt einen Pipeline-Schritt aus.

    Returns:
        True bei Erfolg, False bei Fehler
    """
    step = PIPELINE_STEPS[step_id]
    script_path = SCRIPTS_DIR / step["script"]

    if not script_path.exists():
        print(f"  FEHLER: Script nicht gefunden: {script_path}")
        if logger:
            logger.error(f"Script nicht gefunden: {script_path}")
        return False

    print(f"\n{'=' * 79}")
    print(f"  SCHRITT: {step['name']}")
    print(f"  Script:  {step['script']}")
    print(f"  {step['description']}")
    print(f"{'=' * 79}\n")

    if logger:
        logger.info(f"Starte Schritt: {step_id} ({step['script']})")

    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    try:
        result = subprocess.run(
            cmd,
            cwd=str(SCRIPTS_DIR.parent),
            timeout=None,  # Kein Timeout — Embedding-Extraktion kann >2h dauern
        )

        if result.returncode == 0:
            print(f"\n  -> {step['name']} erfolgreich abgeschlossen.")
            if logger:
                logger.info(f"Schritt {step_id} erfolgreich (returncode=0)")
            return True
        else:
            print(f"\n  -> FEHLER: {step['name']} mit Code {result.returncode} beendet.")
            if logger:
                logger.error(f"Schritt {step_id} fehlgeschlagen (returncode={result.returncode})")
            return False
    except Exception as e:
        print(f"\n  -> FEHLER: {e}")
        if logger:
            logger.error(f"Schritt {step_id} Exception: {e}")
        return False


def run_pipeline(steps: list[str], step_args: dict[str, list[str]] = None) -> dict:
    """
    Fuehrt mehrere Pipeline-Schritte sequentiell aus.

    Returns:
        Dict mit Ergebnissen pro Schritt
    """
    if step_args is None:
        step_args = {}

    results = {}
    start_time = datetime.now()

    print(f"\n{'#' * 79}")
    print(f"  SPOTILYZER TRAINING PIPELINE")
    print(f"  Schritte: {', '.join(steps)}")
    print(f"  Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 79}")

    if logger:
        logger.info(f"Pipeline gestartet: {', '.join(steps)}")

    for step_id in steps:
        if step_id not in PIPELINE_STEPS:
            print(f"  Unbekannter Schritt: {step_id}")
            results[step_id] = False
            continue

        extra_args = step_args.get(step_id, [])
        success = run_step(step_id, extra_args)
        results[step_id] = success

        if not success:
            print(f"\n  Pipeline abgebrochen bei Schritt: {step_id}")
            if logger:
                logger.error(f"Pipeline abgebrochen bei: {step_id}")
            break

    end_time = datetime.now()
    duration = end_time - start_time

    # Summary
    print(f"\n{'#' * 79}")
    print(f"  PIPELINE SUMMARY")
    print(f"{'#' * 79}")
    print(f"  Dauer: {duration}")

    for step_id in steps:
        status = "OK" if results.get(step_id) else "FAIL" if step_id in results else "SKIP"
        name = PIPELINE_STEPS.get(step_id, {}).get("name", step_id)
        print(f"    [{status:4}] {name}")

    if logger:
        logger.info(f"Pipeline beendet: {results}, Dauer: {duration}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE-STATUS
# ══════════════════════════════════════════════════════════════════════════════

def check_pipeline_status(paths: dict, model_key: str = None):
    """Zeigt den aktuellen Status der Pipeline-Outputs."""
    metadata_dir = paths.get("metadata")
    previews_dir = paths.get("previews")
    embeddings_base = paths.get("embeddings")
    models_dir = paths.get("models")
    reports_dir = paths.get("reports")

    jsonl_path = get_tracks_jsonl_path(metadata_dir)

    # Embeddings-Unterverzeichnis je nach gewaehltem Modell
    if model_key and model_key in SUPPORTED_MODELS:
        model_short = SUPPORTED_MODELS[model_key].split("/")[-1]
        embeddings_dir = embeddings_base / model_short
        model_label = f"Embeddings [{model_key}]"
    else:
        embeddings_dir = embeddings_base
        model_label = "Embeddings"

    # Neustes Modell suchen
    model_files = sorted(models_dir.glob("spotilyzer_model_*.joblib"),
                         key=lambda p: p.stat().st_mtime, reverse=True) if models_dir.exists() else []
    newest_model = model_files[0] if model_files else models_dir / "spotilyzer_model.joblib"

    print(f"\n{'=' * 79}")
    print(f"  PIPELINE STATUS  (Embedder: {model_key or 'auto'})")
    print(f"{'=' * 79}")

    checks = [
        ("tracks.jsonl", jsonl_path),
        ("Previews (MP3s)", previews_dir),
        (f"{model_label}/embeddings.npy", embeddings_dir / "embeddings.npy"),
        (f"{model_label}/embeddings_meta.csv", embeddings_dir / "embeddings_meta.csv"),
        (f"Modell ({newest_model.name})", newest_model),
        ("training_report.json", reports_dir / "training_report.json"),
        ("evaluation_report.json", reports_dir / "evaluation_report.json"),
    ]

    for name, path in checks:
        if path.is_dir():
            # Zaehle MP3s in Shard-Verzeichnissen
            mp3_count = 0
            total_size = 0
            if path.exists():
                for mp3 in path.rglob("*.mp3"):
                    mp3_count += 1
                    total_size += mp3.stat().st_size

            if mp3_count > 0:
                total_mb = total_size / (1024*1024)
                print(f"    [OK  ] {name:35} ({mp3_count} Dateien, {total_mb:.0f} MB)")
            else:
                print(f"    [LEER] {name:35} (keine MP3s)")
        elif path.exists():
            size = path.stat().st_size
            if size > 1024 * 1024:
                size_str = f"{size / (1024*1024):.1f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} B"

            # Fuer JSONL: Zeilen zaehlen
            extra = ""
            if path.suffix == ".jsonl":
                with open(path, "r") as f:
                    line_count = sum(1 for line in f if line.strip())
                extra = f", {line_count} Tracks"

            print(f"    [OK  ] {name:35} ({size_str}{extra})")
        else:
            print(f"    [FEHLT] {name:35}")

    print()


# ══════════════════════════════════════════════════════════════════════════════
# INTERAKTIVES MENUE
# ══════════════════════════════════════════════════════════════════════════════

def interactive_menu(paths: dict, initial_model_key: str = None, initial_validated_only: bool = False):
    """Zeigt interaktives Menue fuer Pipeline-Steuerung."""

    selected_model = initial_model_key or _default_model_key()
    validated_only = initial_validated_only

    # Basis-Argumente (modellunabhaengig)
    base_step_args = {
        "scout": ["--charts", "DE", "US", "UK", "FR", "BR", "ES", "GLOBAL"],
    }

    while True:
        model_name = SUPPORTED_MODELS[selected_model]
        step_args = build_model_step_args(selected_model, base_step_args, validated_only=validated_only)

        val_label = "  [validated-only]" if validated_only else ""

        print(f"\n{'=' * 79}")
        print(f"  SPOTILYZER TRAINING PIPELINE")
        print(f"  Embedder: {selected_model}  ({model_name}){val_label}")
        print(f"{'=' * 79}")
        print()
        print(f"  Pipeline-Schritte:")
        print(f"    1. Scout       - Deezer-Scouting (Genre-Cluster + Charts)")
        print(f"    2. Download    - Preview-MP3s herunterladen (MD5-Sharding)")
        print(f"    3. Enrich      - Last.fm-Enrichment")
        print(f"    4. Labels      - Multi-Source-Label-Berechnung")
        print(f"    5. Embeddings  - MERT-Embedding-Extraktion  [{selected_model}]")
        print(f"    6. Train       - XGBoost-Training  [{selected_model}]{val_label}")
        print(f"    7. Evaluate    - Modell-Evaluation  [{selected_model}]{val_label}")
        print()
        print(f"  Kombinationen:")
        print(f"    F. Full Pipeline   (alle Schritte)  [{selected_model}]{val_label}")
        print(f"    T. Train Only      (Embeddings + Train + Evaluate)  [{selected_model}]{val_label}")
        print(f"    E. Enrich + Train  (Enrich + Labels + T)  [{selected_model}]{val_label}")
        print()
        print(f"  Konfiguration:")
        model_opts = "  /  ".join(
            f"[{k}]" if k == selected_model else k
            for k in SUPPORTED_MODELS
        )
        print(f"    M. Embedder wechseln  ({model_opts})")
        val_status = "AN" if validated_only else "AUS"
        print(f"    V. Validated-only  [{val_status}]  (aktuell: {'nur validated' if validated_only else 'alle Labels'})")
        print()
        print(f"  Sonstiges:")
        print(f"    S. Status anzeigen")
        print(f"    Q. Beenden")
        print()

        choice = input("  Auswahl: ").strip().upper()

        if choice == "Q":
            print("  Beendet.")
            break
        elif choice == "M":
            keys = list(SUPPORTED_MODELS.keys())
            selected_model = keys[(keys.index(selected_model) + 1) % len(keys)]
            print(f"  Embedder gewechselt zu: {selected_model}  ({SUPPORTED_MODELS[selected_model]})")
        elif choice == "V":
            validated_only = not validated_only
            status = "aktiviert" if validated_only else "deaktiviert"
            print(f"  Validated-only {status}.")
        elif choice == "S":
            check_pipeline_status(paths, model_key=selected_model)
        elif choice == "1":
            run_step("scout", step_args.get("scout"))
        elif choice == "2":
            run_step("download")
        elif choice == "3":
            run_step("enrich")
        elif choice == "4":
            run_step("labels")
        elif choice == "5":
            run_step("embeddings", step_args.get("embeddings"))
        elif choice == "6":
            run_step("train", step_args.get("train"))
        elif choice == "7":
            run_step("evaluate", step_args.get("evaluate"))
        elif choice == "F":
            run_pipeline(PIPELINE_ORDER, step_args=step_args)
        elif choice == "T":
            run_pipeline(
                ["embeddings", "train", "evaluate"],
                step_args=step_args,
            )
        elif choice == "E":
            run_pipeline(
                ["enrich", "labels", "embeddings", "train", "evaluate"],
                step_args=step_args,
            )
        else:
            print(f"  Unbekannte Auswahl: {choice}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global logger

    paths = load_paths_config()

    parser = argparse.ArgumentParser(
        description="SpotilyzerTraining Pipeline-Orchestrierung"
    )
    parser.add_argument(
        "--model", "-m",
        choices=list(SUPPORTED_MODELS.keys()),
        default=None,
        help=f"Embedder-Modell ({' | '.join(SUPPORTED_MODELS.keys())}, default: aus training.yaml)"
    )
    parser.add_argument("--full", action="store_true", help="Komplette Pipeline ausfuehren")
    parser.add_argument("--scout", action="store_true", help="Nur Deezer-Scouting")
    parser.add_argument("--download", action="store_true", help="Nur Preview-Download")
    parser.add_argument("--enrich", action="store_true", help="Nur Last.fm-Enrichment")
    parser.add_argument("--labels", action="store_true", help="Nur Label-Berechnung")
    parser.add_argument("--embeddings", action="store_true", help="Nur Embedding-Extraktion")
    parser.add_argument("--train", action="store_true", help="Embeddings + Train + Evaluate")
    parser.add_argument("--evaluate", action="store_true", help="Nur Evaluation")
    parser.add_argument("--status", action="store_true", help="Pipeline-Status anzeigen")
    parser.add_argument(
        "--validated-only",
        action="store_true",
        help="Train/Evaluate nur auf validated-Tracks (uebergibt --validated-only an train_model.py + evaluate.py)"
    )
    args = parser.parse_args()

    # Modell-Auswahl: CLI > training.yaml > Fallback 95M
    model_key = args.model or _default_model_key()
    validated_only = args.validated_only

    # Logging
    logger = setup_logging("pipeline", log_dir=paths.get("logs"))

    # Verzeichnisse sicherstellen
    for key in ["metadata", "logs"]:
        if key in paths:
            ensure_dir(paths[key])

    # Basis-Argumente (modellunabhaengig)
    base_step_args = {
        "scout": ["--charts", "DE", "US", "UK", "FR", "BR", "ES", "GLOBAL"],
    }
    step_args = build_model_step_args(model_key, base_step_args, validated_only=validated_only)

    # CLI-Modus
    if args.status:
        check_pipeline_status(paths, model_key=model_key)
        return

    if args.full:
        run_pipeline(PIPELINE_ORDER, step_args=step_args)
        return

    if args.scout:
        run_step("scout", step_args.get("scout"))
        return

    if args.download:
        run_step("download")
        return

    if args.enrich:
        run_step("enrich")
        return

    if args.labels:
        run_step("labels")
        return

    if args.embeddings:
        run_step("embeddings", step_args.get("embeddings"))
        return

    if args.train:
        run_pipeline(
            ["embeddings", "train", "evaluate"],
            step_args=step_args,
        )
        return

    if args.evaluate:
        run_step("evaluate", step_args.get("evaluate"))
        return

    # Kein Flag → interaktives Menue
    interactive_menu(paths, initial_model_key=model_key, initial_validated_only=validated_only)


if __name__ == "__main__":
    main()
