"""
extract_embeddings.py
=====================
Extrahiert MERT-Embeddings aus den heruntergeladenen Preview-MP3s.

MERT-v1-330M (Music undERstanding Transformer) ist ein auf Musik spezialisiertes
Audio-Modell von m-a-p (Music and Audio Processing Lab).
Erzeugt 1024-dimensionale Embeddings (95M: 768-dim).

Input:
  - metadata/tracks.jsonl (Track-Liste mit file_path)
  - previews/{shard}/{track_id}.mp3

Output:
  - outputs/embeddings/embeddings.npy (NumPy-Array)
  - outputs/embeddings/embeddings_meta.csv (Track-IDs, Clusters)

Checkpoint-System:
  - Alle CHECKPOINT_INTERVAL Tracks wird ein Zwischenspeicher geschrieben
  - Bei Neustart wird der Checkpoint automatisch geladen (--resume, default: an)
  - --force ignoriert vorhandenen Checkpoint und startet von vorne
  - --append lädt vorhandenes embeddings.npy + embeddings_meta.csv und verarbeitet nur neue Tracks

GPU-Empfehlung: MERT-330M auf GPU: ~1-2s pro Track, auf CPU: ~20-30s pro Track
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
    load_training_config,
    ensure_dir,
)
from utils.metadata import read_tracks, filter_tracks, get_tracks_jsonl_path

# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Modell-Name wird aus training.yaml (embedder.model) gelesen.
# Fallback falls Config fehlt:
_DEFAULT_MODEL = "m-a-p/MERT-v1-330M"

# Audio-Konfiguration (MERT erwartet 24kHz)
TARGET_SAMPLE_RATE = 24000
MAX_AUDIO_LENGTH_SEC = 30

# Alle N erfolgreichen Tracks einen Checkpoint schreiben
CHECKPOINT_INTERVAL = 500

# GPU-Throttling (Windows: verhindert Display-Treiber-Absturz bei Dauerlast)
# VRAM-Limit: 0.85 → ~5.1 GB auf GTX 1660 Ti (6 GB), lässt ~0.9 GB für Display
GPU_MEMORY_FRACTION = 0.85
# Pause zwischen Tracks in Sekunden (0.05s → ~13 Min Extra bei 16k Tracks)
# Verhindert sustained 100% GPU-Compute-Last → OS/Display bleibt stabil
INTER_TRACK_SLEEP_SEC = 0.05

# Logger (wird in main() initialisiert)
logger = None


# ══════════════════════════════════════════════════════════════════════════════
# MERT-MODELL
# ══════════════════════════════════════════════════════════════════════════════

class MERTEmbedder:
    """Wrapper fuer MERT-Embedding-Extraktion."""

    def __init__(self, model_name: str, device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Lade MERT-Modell auf {self.device.upper()}...")
        if logger:
            logger.info(f"Lade MERT-Modell {model_name} auf {self.device.upper()}")

        # VRAM-Limit setzen (verhindert Display-Treiber-Absturz auf Windows)
        if self.device == "cuda":
            torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
            if logger:
                logger.info(f"VRAM-Limit: {GPU_MEMORY_FRACTION:.0%} (GPU_MEMORY_FRACTION)")

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
            try:
                waveform, sample_rate = torchaudio.load(str(filepath), backend="soundfile")
            except Exception:
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
            embedding = hidden_states.mean(dim=1).squeeze(0)  # [1024]

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
# CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EmbeddingRecord:
    """Metadaten fuer ein Embedding."""
    track_id: int
    clusters: str  # Komma-separiert
    filename: str
    embedding_idx: int


def load_checkpoint(checkpoint_path: Path) -> tuple[list, list, set]:
    """
    Laedt einen vorhandenen Checkpoint.

    Returns:
        (embeddings_list, records_list, done_ids)
    """
    if not checkpoint_path.exists():
        return [], [], set()

    try:
        data = np.load(checkpoint_path, allow_pickle=True)
        embeddings = list(data["embeddings"])
        raw_records = data["records"]  # structured array oder object array

        records = []
        done_ids = set()
        for r in raw_records:
            rec = EmbeddingRecord(
                track_id=int(r["track_id"]),
                clusters=str(r["clusters"]),
                filename=str(r["filename"]),
                embedding_idx=int(r["embedding_idx"]),
            )
            records.append(rec)
            done_ids.add(int(r["track_id"]))

        print(f"  Checkpoint geladen: {len(done_ids)} Tracks bereits verarbeitet")
        if logger:
            logger.info(f"Checkpoint geladen: {len(done_ids)} Tracks")
        return embeddings, records, done_ids

    except Exception as e:
        print(f"  WARNUNG: Checkpoint konnte nicht geladen werden: {e} — starte neu")
        if logger:
            logger.warning(f"Checkpoint-Ladefehler: {e}")
        return [], [], set()


def save_checkpoint(
    checkpoint_path: Path,
    embeddings_list: list,
    records: list,
):
    """Speichert einen Zwischenspeicher."""
    try:
        records_array = np.array(
            [{"track_id": r.track_id, "clusters": r.clusters,
              "filename": r.filename, "embedding_idx": r.embedding_idx}
             for r in records],
            dtype=object,
        )
        np.savez(
            checkpoint_path,
            embeddings=np.stack(embeddings_list, axis=0),
            records=records_array,
        )
        if logger:
            logger.debug(f"Checkpoint gespeichert: {len(records)} Tracks")
    except Exception as e:
        print(f"  WARNUNG: Checkpoint konnte nicht gespeichert werden: {e}")
        if logger:
            logger.warning(f"Checkpoint-Speicherfehler: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# BATCH-VERARBEITUNG
# ══════════════════════════════════════════════════════════════════════════════

def process_batch(
    embedder: MERTEmbedder,
    tracks: list[dict],
    data_root: Path,
    output_dir: Path,
    checkpoint_embeddings: list = None,
    checkpoint_records: list = None,
    checkpoint_path: Path = None,
    existing_embeddings: list = None,
    existing_records: list = None,
) -> dict:
    """
    Verarbeitet Tracks anhand JSONL-Metadaten.

    checkpoint_embeddings / checkpoint_records: bereits verarbeitete Daten aus
    einem frueheren (abgebrochenen) Lauf — werden mit neuen Ergebnissen zusammengefuehrt.
    existing_embeddings / existing_records: fertige Embeddings aus --append-Modus.
    """
    checkpoint_embeddings = checkpoint_embeddings or []
    checkpoint_records = checkpoint_records or []
    existing_embeddings = existing_embeddings or []
    existing_records = existing_records or []

    if not tracks:
        total_existing = len(existing_embeddings) + len(checkpoint_embeddings)
        if total_existing:
            print(f"  Keine neuen Tracks — {total_existing} bereits verarbeitet "
                  f"({len(existing_embeddings)} Append + {len(checkpoint_embeddings)} Checkpoint).")
        else:
            print("  Keine Tracks mit file_path gefunden!")
            if logger:
                logger.warning("Keine Tracks mit file_path")
        return {"success": 0, "failed": 0}

    already_done = len(existing_embeddings) + len(checkpoint_embeddings)
    print(f"  Verarbeite {len(tracks)} neue Dateien"
          + (f" (+ {already_done} bereits verarbeitet)" if already_done else "") + "...")
    if logger:
        logger.info(f"Verarbeite {len(tracks)} neue Tracks ({already_done} bereits verarbeitet)")

    # Laufende Listen (starten mit Append-Daten, dann Checkpoint drüber)
    # Beide sind jetzt EmbeddingRecord-Objekte → asdict() funktioniert einheitlich
    embeddings_list = list(existing_embeddings) + list(checkpoint_embeddings)
    records = list(existing_records) + list(checkpoint_records)

    # Embedding-Indizes korrigieren (Checkpoint hatte 0-basiert aufgebaut)
    for i, rec in enumerate(records):
        rec.embedding_idx = i

    stats = {"success": 0, "failed": 0, "errors": []}
    start_time = time.time()
    next_checkpoint_at = already_done + CHECKPOINT_INTERVAL

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

        # GPU-Throttling: kurze Pause damit Display-Treiber Luft bekommt
        if INTER_TRACK_SLEEP_SEC > 0:
            time.sleep(INTER_TRACK_SLEEP_SEC)

        # Checkpoint alle CHECKPOINT_INTERVAL erfolgreiche Tracks
        total_done = already_done + stats["success"]
        if checkpoint_path and total_done >= next_checkpoint_at:
            save_checkpoint(checkpoint_path, embeddings_list, records)
            print(f"\n  [Checkpoint] {total_done} Tracks gespeichert.")
            next_checkpoint_at = total_done + CHECKPOINT_INTERVAL

    elapsed = time.time() - start_time

    # Finales Speichern
    if embeddings_list:
        ensure_dir(output_dir)

        embeddings_array = np.stack(embeddings_list, axis=0)
        embeddings_path = output_dir / "embeddings.npy"
        np.save(embeddings_path, embeddings_array)

        records_df = pd.DataFrame([asdict(r) for r in records])
        records_path = output_dir / "embeddings_meta.csv"
        records_df.to_csv(records_path, index=False)

        info = {
            "model": embedder.model_name,
            "embedding_dim": int(embeddings_array.shape[1]),
            "num_embeddings": int(embeddings_array.shape[0]),
            "sample_rate": TARGET_SAMPLE_RATE,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_time_sec": round(elapsed, 1),
            "avg_time_per_file_sec": round(elapsed / max(len(tracks), 1), 2),
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

        # Checkpoint loeschen nach erfolgreichem Abschluss
        if checkpoint_path and checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"  Checkpoint geloescht (nicht mehr benoetigt).")

    stats["elapsed"] = elapsed
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global logger

    paths = load_paths_config()
    training_cfg = load_training_config()

    data_root = paths.get("data_root")
    metadata_dir = paths.get("metadata")
    jsonl_path = get_tracks_jsonl_path(metadata_dir)
    embeddings_base = paths.get("embeddings", Path("./outputs/embeddings"))

    # Modell aus training.yaml, CLI kann es überschreiben
    cfg_model = training_cfg.get("embedder", {}).get("model", _DEFAULT_MODEL)

    parser = argparse.ArgumentParser(
        description="Extrahiere MERT-Embeddings aus Preview-MP3s"
    )
    parser.add_argument(
        "--model",
        default=cfg_model,
        help=f"HuggingFace-Modell-Name (default: aus training.yaml = {cfg_model})"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output-Verzeichnis (default: outputs/embeddings/<model-short-name>/)"
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
        "--resume",
        action="store_true",
        default=True,
        help="Checkpoint laden und weiterarbeiten (default: an)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Checkpoint ignorieren, von vorne beginnen"
    )
    parser.add_argument(
        "--append",
        action="store_true",
        default=False,
        help="Vorhandenes embeddings.npy laden, nur neue Tracks verarbeiten und zusammenführen"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Modul-Datensatz statt Haupt-JSONL verwenden "
             "(z.B. 'spotify_charts' → datasets/spotify_charts/tracks.jsonl)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Zeige was verarbeitet wuerde, ohne tatsaechlich zu laden"
    )
    args = parser.parse_args()

    # Kurzname-Aliase auflösen (95M → m-a-p/MERT-v1-95M etc.)
    MODEL_ALIASES = {
        "95M":  "m-a-p/MERT-v1-95M",
        "330M": "m-a-p/MERT-v1-330M",
        "MERT-v1-95M":  "m-a-p/MERT-v1-95M",
        "MERT-v1-330M": "m-a-p/MERT-v1-330M",
    }
    model_name = MODEL_ALIASES.get(args.model, args.model)
    model_short = model_name.split("/")[-1]  # "MERT-v1-95M"

    # Output-Verzeichnis: explizit oder automatisch aus Modell-Kurzname ableiten
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(embeddings_base) / model_short

    # Logging
    logger = setup_logging("embeddings", log_dir=paths.get("logs"))

    print(f"{'=' * 79}")
    print(f"  MERT EMBEDDING EXTRACTOR")
    print(f"{'=' * 79}")
    print(f"  Modell:    {model_name}")

    # JSONL-Pfad: Haupt-Datensatz oder Modul-Datensatz
    if args.dataset:
        datasets_dir = paths.get("datasets", paths["data_root"] / "datasets")
        jsonl_path = datasets_dir / args.dataset / "tracks.jsonl"

    checkpoint_path = output_dir / "embeddings_checkpoint.npz"

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

    # Checkpoint laden (wenn --resume und nicht --force)
    checkpoint_embeddings, checkpoint_records, done_ids = [], [], set()
    if args.resume and not args.force:
        checkpoint_embeddings, checkpoint_records, done_ids = load_checkpoint(checkpoint_path)

    # --append: vorhandenes embeddings.npy + meta.csv laden, bereits verarbeitete IDs übernehmen
    existing_embeddings, existing_records = [], []
    if args.append and not args.force:
        existing_meta_path = output_dir / "embeddings_meta.csv"
        existing_npy_path = output_dir / "embeddings.npy"
        if existing_meta_path.exists() and existing_npy_path.exists():
            existing_arr = np.load(existing_npy_path)
            existing_df = pd.read_csv(existing_meta_path)
            existing_ids = set(existing_df["track_id"].tolist())
            # Nur übernehmen wenn Arrays konsistent
            if len(existing_arr) == len(existing_df):
                existing_embeddings = [existing_arr[i] for i in range(len(existing_arr))]
                # Dicts → EmbeddingRecord konvertieren für einheitliche Weiterverarbeitung
                existing_records = [
                    EmbeddingRecord(
                        track_id=int(row["track_id"]),
                        clusters=str(row.get("clusters", "")),
                        filename=str(row.get("filename", "")),
                        embedding_idx=int(row.get("embedding_idx", i)),
                    )
                    for i, row in existing_df.iterrows()
                ]
                done_ids = done_ids | existing_ids
                print(f"  Append-Modus: {len(existing_ids)} bereits eingebettete Tracks geladen")
            else:
                print(f"  Warnung: embeddings.npy ({len(existing_arr)}) und meta.csv "
                      f"({len(existing_df)}) inkonsistent — Append deaktiviert")
        else:
            print(f"  Append-Modus: keine vorhandenen Embeddings gefunden, starte neu")

    # Bereits verarbeitete Tracks herausfiltern
    if done_ids:
        before = len(tracks)
        tracks = [t for t in tracks if t["track_id"] not in done_ids]
        print(f"  Ueberspringe: {before - len(tracks)} bereits verarbeitete Tracks")

    print(f"  JSONL:     {jsonl_path}")
    print(f"  Tracks:    {len(tracks)} verbleibend"
          + (f" ({len(done_ids)} aus Checkpoint)" if done_ids else ""))
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
        f"Embedding-Extraktion gestartet: {len(tracks)} neue Tracks, "
        f"{len(done_ids)} aus Checkpoint, Device: {actual_device.upper()}"
    )

    # Geschaetzte Zeit (95M: ~0.5s/Track GPU; 330M: ~0.8s/Track GPU)
    is_330m = "330M" in model_name
    time_per_file = (0.8 if is_330m else 0.5) if actual_device == "cuda" else 12.0
    est_minutes = len(tracks) * time_per_file / 60
    print(f"\n  Geschaetzte Zeit: ~{est_minutes:.0f} Minuten")
    print(f"  Checkpoint:  alle {CHECKPOINT_INTERVAL} Tracks")

    if args.dry_run:
        print(f"\n  [DRY RUN] Keine Verarbeitung durchgefuehrt.")
        return

    # Modell laden und verarbeiten
    print()
    embedder = MERTEmbedder(model_name, device)
    print()

    stats = process_batch(
        embedder,
        tracks,
        data_root,
        output_dir,
        checkpoint_embeddings=checkpoint_embeddings,
        checkpoint_records=checkpoint_records,
        checkpoint_path=checkpoint_path,
        existing_embeddings=existing_embeddings,
        existing_records=existing_records,
    )

    # Ergebnis
    total_success = stats["success"] + len(done_ids)
    print(f"\n{'─' * 79}")
    print(f"  EXTRACTION COMPLETE")
    print(f"{'─' * 79}")
    print(f"  Erfolgreich:    {total_success:>5}  (davon neu: {stats['success']}, Checkpoint: {len(done_ids)})")
    print(f"  Fehlgeschlagen: {stats['failed']:>5}")
    print(f"  Zeit (neu):     {stats.get('elapsed', 0):.1f}s ({stats.get('elapsed', 0)/60:.1f} min)")

    if stats['success'] > 0:
        print(f"  Durchschnitt:   {stats.get('elapsed', 0)/stats['success']:.2f}s pro Datei")

    if stats.get("errors"):
        print(f"\n  Fehler (erste 5):")
        for filename, error in stats["errors"][:5]:
            print(f"    {filename}: {error}")

    logger.info(
        f"Extraktion abgeschlossen: {total_success} gesamt "
        f"({stats['success']} neu, {len(done_ids)} Checkpoint), "
        f"{stats['failed']} failed, {stats.get('elapsed', 0):.1f}s"
    )


if __name__ == "__main__":
    main()
