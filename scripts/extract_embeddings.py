"""
extract_embeddings.py
=====================
Extrahiert MERT-Embeddings aus den heruntergeladenen Preview-MP3s.

MERT (Music undERstanding Transformer) ist ein auf Musik spezialisiertes
Audio-Modell von m-a-p (Music and Audio Processing Lab).

Input:
  - metadata/tracks.jsonl (Track-Liste mit file_path)
  - previews/{shard}/{track_id}.mp3

Output:
  - outputs/embeddings/embeddings.npy (NumPy-Array)
  - outputs/embeddings/embeddings_meta.csv (Track-IDs, Clusters)

GPU-Empfehlung: MERT auf GPU: <1s pro Track, auf CPU: ~10-15s pro Track
"""

import sys
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from _utils import (
    setup_logging,
    load_paths_config,
    ensure_dir,
)
from utils.metadata import read_tracks, filter_tracks, get_tracks_jsonl_path

# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

MODEL_NAME = "m-a-p/MERT-v1-95M"  # ~380MB, musik-optimiert

# Audio-Konfiguration (MERT erwartet 24kHz)
TARGET_SAMPLE_RATE = 24000
MAX_AUDIO_LENGTH_SEC = 30

# Logger (wird in main() initialisiert)
logger = None


# ══════════════════════════════════════════════════════════════════════════════
# MERT-MODELL
# ══════════════════════════════════════════════════════════════════════════════

class MERTEmbedder:
    """Wrapper fuer MERT-Embedding-Extraktion."""

    def __init__(self, model_name: str = MODEL_NAME, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Lade MERT-Modell auf {self.device.upper()}...")
        if logger:
            logger.info(f"Lade MERT-Modell {model_name} auf {self.device.upper()}")

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

        print(f"  Modell geladen: {model_name}")
        if logger:
            logger.info(f"Modell geladen: {model_name}")

    def load_audio(self, filepath: Path) -> torch.Tensor | None:
        """Laedt und preprocessed eine Audio-Datei."""
        try:
            waveform, sample_rate = torchaudio.load(filepath)

            # Mono konvertieren falls Stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample auf 24kHz falls noetig
            if sample_rate != TARGET_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=TARGET_SAMPLE_RATE
                )
                waveform = resampler(waveform)

            # Auf maximale Laenge begrenzen
            max_samples = TARGET_SAMPLE_RATE * MAX_AUDIO_LENGTH_SEC
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]

            return waveform.squeeze(0)  # [samples]

        except Exception as e:
            if logger:
                logger.error(f"Fehler beim Laden von {filepath.name}: {e}")
            return None

    @torch.no_grad()
    def extract_embedding(self, waveform: torch.Tensor) -> np.ndarray | None:
        """Extrahiert das Embedding fuer eine Waveform."""
        try:
            inputs = self.processor(
                waveform.numpy(),
                sampling_rate=TARGET_SAMPLE_RATE,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs, output_hidden_states=True)

            # Letzten Hidden State nehmen und ueber Zeit mitteln
            hidden_states = outputs.last_hidden_state
            embedding = hidden_states.mean(dim=1).squeeze(0)  # [768]

            return embedding.cpu().numpy()

        except Exception as e:
            if logger:
                logger.error(f"Fehler bei Embedding-Extraktion: {e}")
            return None

    def process_file(self, filepath: Path) -> np.ndarray | None:
        """Kompletter Pipeline: Laden -> Embedding extrahieren."""
        waveform = self.load_audio(filepath)
        if waveform is None:
            return None
        return self.extract_embedding(waveform)


# ══════════════════════════════════════════════════════════════════════════════
# BATCH-VERARBEITUNG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EmbeddingRecord:
    """Metadaten fuer ein Embedding."""
    track_id: int
    clusters: str  # Komma-separiert
    filename: str
    embedding_idx: int


def process_batch(
    embedder: MERTEmbedder,
    tracks: list[dict],
    data_root: Path,
    output_dir: Path,
) -> dict:
    """Verarbeitet Tracks anhand JSONL-Metadaten."""

    if not tracks:
        print("  Keine Tracks mit file_path gefunden!")
        if logger:
            logger.warning("Keine Tracks mit file_path")
        return {"success": 0, "failed": 0}

    print(f"  Verarbeite {len(tracks)} Dateien...")
    if logger:
        logger.info(f"Verarbeite {len(tracks)} Tracks")

    embeddings_list = []
    records = []
    stats = {"success": 0, "failed": 0, "errors": []}

    start_time = time.time()

    for track in tqdm(tracks, desc="Extracting", unit="file"):
        track_id = track["track_id"]
        file_path = track.get("file_path", "")

        # Absoluten Pfad aufloesen
        abs_path = data_root / file_path
        if not abs_path.exists():
            stats["failed"] += 1
            stats["errors"].append((str(track_id), "file not found"))
            if logger:
                logger.warning(f"Datei nicht gefunden: {abs_path}")
            continue

        embedding = embedder.process_file(abs_path)

        if embedding is None:
            stats["failed"] += 1
            stats["errors"].append((str(track_id), "extraction failed"))
            if logger:
                logger.warning(f"Extraction fehlgeschlagen: {abs_path.name}")
            continue

        embeddings_list.append(embedding)
        records.append(EmbeddingRecord(
            track_id=track_id,
            clusters=",".join(track.get("clusters", [])),
            filename=abs_path.name,
            embedding_idx=len(embeddings_list) - 1,
        ))
        stats["success"] += 1

    elapsed = time.time() - start_time

    # Speichern
    if embeddings_list:
        ensure_dir(output_dir)

        embeddings_array = np.stack(embeddings_list, axis=0)
        embeddings_path = output_dir / "embeddings.npy"
        np.save(embeddings_path, embeddings_array)

        records_df = pd.DataFrame([asdict(r) for r in records])
        records_path = output_dir / "embeddings_meta.csv"
        records_df.to_csv(records_path, index=False)

        info = {
            "model": MODEL_NAME,
            "embedding_dim": int(embeddings_array.shape[1]),
            "num_embeddings": int(embeddings_array.shape[0]),
            "sample_rate": TARGET_SAMPLE_RATE,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_time_sec": round(elapsed, 1),
            "avg_time_per_file_sec": round(elapsed / len(tracks), 2),
        }
        info_path = output_dir / "embeddings_info.json"
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

        print(f"\n  Gespeichert:")
        print(f"    {embeddings_path} ({embeddings_array.shape})")
        print(f"    {records_path}")
        print(f"    {info_path}")

        if logger:
            logger.info(
                f"Embeddings gespeichert: {embeddings_array.shape}, "
                f"Dauer: {elapsed:.1f}s"
            )

    stats["elapsed"] = elapsed
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global logger

    paths = load_paths_config()

    data_root = paths.get("data_root")
    metadata_dir = paths.get("metadata")
    jsonl_path = get_tracks_jsonl_path(metadata_dir)
    default_output = str(paths.get("embeddings", "./outputs/embeddings"))

    parser = argparse.ArgumentParser(
        description="Extrahiere MERT-Embeddings aus Preview-MP3s"
    )
    parser.add_argument(
        "--output", "-o",
        default=default_output,
        help=f"Output-Verzeichnis fuer Embeddings (default: {default_output})"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximale Anzahl zu verarbeiten (0 = alle)"
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default=None,
        help="Nur bestimmten Cluster verarbeiten"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device fuer MERT (default: auto)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Zeige was verarbeitet wuerde, ohne tatsaechlich zu laden"
    )
    args = parser.parse_args()

    # Logging
    logger = setup_logging("embeddings", log_dir=paths.get("logs"))

    print(f"{'=' * 79}")
    print(f"  MERT EMBEDDING EXTRACTOR")
    print(f"{'=' * 79}")

    output_dir = Path(args.output)

    if not jsonl_path.exists():
        print(f"  Fehler: tracks.jsonl nicht gefunden: {jsonl_path}")
        logger.error(f"tracks.jsonl nicht gefunden: {jsonl_path}")
        sys.exit(1)

    # Tracks laden und filtern (nur mit file_path)
    all_tracks = read_tracks(jsonl_path)
    tracks = filter_tracks(all_tracks, has_file=True)

    if args.cluster:
        tracks = filter_tracks(tracks, cluster=args.cluster)

    if args.limit > 0:
        tracks = tracks[:args.limit]

    print(f"  JSONL:     {jsonl_path}")
    print(f"  Tracks:    {len(tracks)} (mit file_path)")
    print(f"  Output:    {output_dir}")

    if args.cluster:
        print(f"  Filter:    cluster={args.cluster}")
    if args.limit > 0:
        print(f"  Limit:     {args.limit}")

    # Device
    device = args.device if args.device != "auto" else None
    actual_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device:    {actual_device.upper()}")

    if actual_device == "cuda":
        print(f"  GPU:       {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:      {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    logger.info(
        f"Embedding-Extraktion gestartet: {len(tracks)} Tracks, "
        f"Device: {actual_device.upper()}"
    )

    # Geschaetzte Zeit
    time_per_file = 0.8 if actual_device == "cuda" else 12.0
    est_minutes = len(tracks) * time_per_file / 60
    print(f"\n  Geschaetzte Zeit: ~{est_minutes:.0f} Minuten")

    if args.dry_run:
        print(f"\n  [DRY RUN] Keine Verarbeitung durchgefuehrt.")
        return

    # Modell laden und verarbeiten
    print()
    embedder = MERTEmbedder(MODEL_NAME, device)
    print()

    stats = process_batch(
        embedder, tracks, data_root, output_dir,
    )

    # Ergebnis
    print(f"\n{'─' * 79}")
    print(f"  EXTRACTION COMPLETE")
    print(f"{'─' * 79}")
    print(f"  Erfolgreich:    {stats['success']:>5}")
    print(f"  Fehlgeschlagen: {stats['failed']:>5}")
    print(f"  Zeit:           {stats.get('elapsed', 0):.1f}s ({stats.get('elapsed', 0)/60:.1f} min)")

    if stats['success'] > 0:
        print(f"  Durchschnitt:   {stats.get('elapsed', 0)/stats['success']:.2f}s pro Datei")

    if stats.get("errors"):
        print(f"\n  Fehler (erste 5):")
        for filename, error in stats["errors"][:5]:
            print(f"    {filename}: {error}")

    logger.info(
        f"Extraktion abgeschlossen: {stats['success']} ok, "
        f"{stats['failed']} failed, {stats.get('elapsed', 0):.1f}s"
    )


if __name__ == "__main__":
    main()
