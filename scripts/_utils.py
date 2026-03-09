"""
_utils.py
=========
Gemeinsame Hilfsfunktionen für alle Pipeline-Skripte.

- Logging-Setup (datierte Log-Dateien)
- YAML-Config-Laden (paths, thresholds, training, clusters)
- Pfad-Resolution
"""

import logging
import sys
from datetime import date
from pathlib import Path

import yaml


# Projekt-Root (eine Ebene über scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def setup_logging(script_name: str, log_dir: Path = None) -> logging.Logger:
    """
    Richtet Logging für ein Skript ein.

    - Schreibt nach logs/{script_name}_YYYY-MM-DD.log
    - Gibt gleichzeitig auf stderr aus

    Args:
        script_name: Name des Skripts (ohne .py)
        log_dir: Log-Verzeichnis (default: PROJECT_ROOT/logs)

    Returns:
        Konfigurierter Logger
    """
    if log_dir is None:
        log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{script_name}_{date.today().isoformat()}.log"

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)

    # Datei-Handler (detailliert)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    ))

    # Konsolen-Handler (kürzer)
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"=== {script_name} gestartet ===")

    return logger


def load_yaml(config_path: Path) -> dict:
    """Lädt eine YAML-Config-Datei."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_paths_config(config_path: Path = None) -> dict:
    """
    Lädt paths.yaml und löst relative Pfade auf.

    Returns:
        Dict mit aufgelösten Path-Objekten
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "paths.yaml"

    raw = load_yaml(config_path)
    paths = raw.get("paths", raw)

    resolved = {}
    for key, value in paths.items():
        p = Path(value)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        resolved[key] = p

    return resolved


def load_thresholds_config(config_path: Path = None) -> dict:
    """Lädt thresholds.yaml."""
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "thresholds.yaml"
    return load_yaml(config_path)


def load_training_config(config_path: Path = None) -> dict:
    """Lädt training.yaml."""
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "training.yaml"
    return load_yaml(config_path)


def load_clusters_config(config_path: Path = None) -> dict:
    """
    Lädt clusters.yaml.

    Returns:
        Vollständiges Dict aus clusters.yaml.
        Genre-Cluster haben keys: display_name, description, seed_artists.
        Spezial-Keys: "charts", "scouting".
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "clusters.yaml"
    return load_yaml(config_path)


def get_genre_clusters(clusters_config: dict) -> dict:
    """
    Extrahiert nur die Genre-Cluster (ohne charts, scouting).

    Args:
        clusters_config: Vollständiges clusters.yaml dict

    Returns:
        Dict mit nur Genre-Cluster-Definitionen
    """
    skip_keys = {"charts", "scouting"}
    return {k: v for k, v in clusters_config.items() if k not in skip_keys}


def get_charts_config(clusters_config: dict) -> dict:
    """Extrahiert die Charts-Konfiguration aus clusters.yaml."""
    return clusters_config.get("charts", {})


def get_scouting_config(clusters_config: dict) -> dict:
    """Extrahiert die Scouting-Konfiguration aus clusters.yaml."""
    return clusters_config.get("scouting", {})


def ensure_dir(path: Path) -> Path:
    """Erstellt Verzeichnis falls nötig und gibt es zurück."""
    path.mkdir(parents=True, exist_ok=True)
    return path
