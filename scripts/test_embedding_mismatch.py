"""
test_embedding_mismatch.py
==========================
Vergleicht die Klassifikationsergebnisse des aktiven Modells zwischen:

  ALT: erste 30s des Tracks → 1 Embedding (bisheriges Trainings-Verfahren)
  NEU: alle 30s-Chunks → Mean-Pool → 1 Embedding (neuer Haupt-Embedder)

Zweck: Messen ob die Embedding-Distribution-Änderung die Vorhersagen
des trainierten XGBoost-Modells messbar beeinflusst, bevor im Hauptprojekt
weitergebaut wird.

Verwendung:
  python scripts/test_embedding_mismatch.py --files <audio1> <audio2> ...
  python scripts/test_embedding_mismatch.py --dir <ordner_mit_audiofiles>
  python scripts/test_embedding_mismatch.py --dir <ordner> --model <pfad_zu_joblib>

Unterstützte Formate: .mp3 .flac .wav .ogg .m4a .aac

Ausgabe:
  - Pro Track: Label (alt/neu), Probabilities (alt/neu), Änderung
  - Gesamt: Übereinstimmungsrate, mittlerer Probability-Shift
"""

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torchaudio

# Projekt-Root zu sys.path hinzufügen damit _utils importierbar ist
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import load_training_config, load_paths_config

# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

TARGET_SAMPLE_RATE = 24000
CHUNK_SEC = 30
MIN_CHUNK_SEC = 5
SUPPORTED_FORMATS = {".mp3", ".flac", ".wav", ".ogg", ".m4a", ".aac"}

_DEFAULT_MODEL = "m-a-p/MERT-v1-330M"


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO LADEN
# ══════════════════════════════════════════════════════════════════════════════

def load_audio(filepath: Path) -> torch.Tensor:
    """Lädt Audio, konvertiert zu Mono, resampled auf 24kHz. Gibt 1D-Tensor zurück."""
    try:
        waveform, sr = torchaudio.load(str(filepath), backend="soundfile")
    except Exception:
        waveform, sr = torchaudio.load(str(filepath))

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)
    return waveform.squeeze(0)  # [samples]


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING-STRATEGIEN
# ══════════════════════════════════════════════════════════════════════════════

def embed_first30s(waveform: torch.Tensor, model, processor, device: str) -> np.ndarray:
    """ALT: erste 30s → 1 Embedding."""
    max_samples = TARGET_SAMPLE_RATE * CHUNK_SEC
    clip = waveform[:max_samples]
    return _extract_single(clip, model, processor, device)


def embed_chunked_mean(waveform: torch.Tensor, model, processor, device: str) -> np.ndarray:
    """NEU: alle 30s-Chunks → Mean-Pool → 1 Embedding."""
    chunks = _chunk_waveform(waveform)
    embeddings = [_extract_single(chunk, model, processor, device) for chunk in chunks]
    return np.mean(embeddings, axis=0)


def _chunk_waveform(waveform: torch.Tensor) -> list[torch.Tensor]:
    """Teilt Waveform in CHUNK_SEC-Segmente. Letzter Chunk wird gepaddet (min MIN_CHUNK_SEC)."""
    chunk_samples = TARGET_SAMPLE_RATE * CHUNK_SEC
    min_samples = TARGET_SAMPLE_RATE * MIN_CHUNK_SEC
    total = waveform.shape[0]
    chunks = []

    for start in range(0, total, chunk_samples):
        chunk = waveform[start : start + chunk_samples]
        if chunk.shape[0] < min_samples:
            break
        if chunk.shape[0] < chunk_samples:
            pad = torch.zeros(chunk_samples - chunk.shape[0], dtype=chunk.dtype)
            chunk = torch.cat([chunk, pad])
        chunks.append(chunk)

    if not chunks:
        pad = torch.zeros(chunk_samples - waveform.shape[0], dtype=waveform.dtype)
        chunks.append(torch.cat([waveform, pad]))

    return chunks


@torch.no_grad()
def _extract_single(waveform: torch.Tensor, model, processor, device: str) -> np.ndarray:
    """Extrahiert ein Embedding aus einem einzelnen Waveform-Chunk."""
    inputs = processor(
        waveform.numpy(),
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs, output_hidden_states=True)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
    return embedding.cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def find_audio_files(directory: Path) -> list[Path]:
    files = []
    for fmt in SUPPORTED_FORMATS:
        files.extend(directory.glob(f"*{fmt}"))
    return sorted(files)


def format_probs(probs: np.ndarray, classes: list[str]) -> str:
    return "  ".join(f"{c}={p:.3f}" for c, p in zip(classes, probs))


def main():
    training_cfg = load_training_config()
    cfg_model = training_cfg.get("embedder", {}).get("model", _DEFAULT_MODEL)

    # Aktives Modell finden (neuestes *kworb*validated*.joblib)
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "outputs" / "models"
    candidate_models = sorted(
        models_dir.glob("spotilyzer_model_MERTv1330M_main+spotify_charts+kworb_validated_*.joblib"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    default_model_path = str(candidate_models[0]) if candidate_models else ""

    parser = argparse.ArgumentParser(
        description="Vergleiche alt (erste 30s) vs. neu (chunked mean-pool) Embeddings"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--files", nargs="+", type=Path, help="Audio-Dateien")
    group.add_argument("--dir", type=Path, help="Ordner mit Audio-Dateien")
    parser.add_argument(
        "--model", type=Path, default=default_model_path,
        help=f"Pfad zum .joblib-Modell (default: {default_model_path or 'nicht gefunden'})"
    )
    parser.add_argument(
        "--mert-model", default=cfg_model,
        help=f"MERT HuggingFace-Name (default: {cfg_model})"
    )
    parser.add_argument(
        "--device", choices=["cuda", "cpu", "auto"], default="auto"
    )
    args = parser.parse_args()

    # Audio-Dateien sammeln
    if args.files:
        audio_files = [p for p in args.files if p.suffix.lower() in SUPPORTED_FORMATS]
    else:
        audio_files = find_audio_files(args.dir)

    if not audio_files:
        print("Keine unterstützten Audio-Dateien gefunden.")
        sys.exit(1)

    if not args.model or not Path(args.model).exists():
        print(f"Modell nicht gefunden: {args.model}")
        print("Bitte --model <pfad> angeben.")
        sys.exit(1)

    # Device
    device = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"{'=' * 70}")
    print(f"  EMBEDDING MISMATCH TEST")
    print(f"{'=' * 70}")
    print(f"  Tracks:      {len(audio_files)}")
    print(f"  MERT-Modell: {args.mert_model}")
    print(f"  XGB-Modell:  {Path(args.model).name}")
    print(f"  Device:      {device.upper()}")
    print()

    # XGBoost-Modell laden
    print("  Lade XGBoost-Modell...", end=" ", flush=True)
    bundle = joblib.load(args.model)
    xgb_model = bundle["model"]
    label_enc = bundle["label_encoder"]
    classes = list(label_enc.classes_)
    print(f"OK  (Klassen: {classes})")

    # MERT laden
    from transformers import AutoModel, AutoProcessor
    print("  Lade MERT-Modell (kann ~30s dauern)...")
    processor = AutoProcessor.from_pretrained(args.mert_model, trust_remote_code=True)
    mert = AutoModel.from_pretrained(args.mert_model, trust_remote_code=True)
    mert.to(device)
    mert.eval()
    print("  MERT geladen.")
    print()

    # Pro-Track-Ergebnisse
    results = []
    print(f"  {'Track':<35}  {'Dauer':>6}  {'ALT':>5}  {'NEU':>5}  {'Übereins.':>9}")
    print(f"  {'-'*35}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*9}")

    for filepath in audio_files:
        try:
            waveform = load_audio(filepath)
        except Exception as e:
            print(f"  [FEHLER] {filepath.name}: {e}")
            continue

        duration_sec = waveform.shape[0] / TARGET_SAMPLE_RATE
        n_chunks = max(1, int(duration_sec // CHUNK_SEC))

        # ALT: erste 30s
        emb_old = embed_first30s(waveform, mert, processor, device)
        probs_old = xgb_model.predict_proba([emb_old])[0]
        label_old = classes[np.argmax(probs_old)]

        # NEU: chunked mean-pool
        emb_new = embed_chunked_mean(waveform, mert, processor, device)
        probs_new = xgb_model.predict_proba([emb_new])[0]
        label_new = classes[np.argmax(probs_new)]

        match = label_old == label_new
        prob_shift = float(np.mean(np.abs(probs_new - probs_old)))

        results.append({
            "file": filepath.name,
            "duration_sec": duration_sec,
            "n_chunks": n_chunks,
            "label_old": label_old,
            "label_new": label_new,
            "probs_old": probs_old,
            "probs_new": probs_new,
            "match": match,
            "prob_shift": prob_shift,
        })

        match_str = "✓" if match else f"✗ ({label_old}→{label_new})"
        name_short = filepath.stem[:35]
        print(f"  {name_short:<35}  {duration_sec:>5.0f}s  {label_old:>5}  {label_new:>5}  {match_str:>9}")

    if not results:
        print("Keine Tracks erfolgreich verarbeitet.")
        sys.exit(1)

    # Gesamtauswertung
    n = len(results)
    n_match = sum(r["match"] for r in results)
    mean_shift = float(np.mean([r["prob_shift"] for r in results]))
    max_shift = float(np.max([r["prob_shift"] for r in results]))
    mismatches = [r for r in results if not r["match"]]

    print()
    print(f"{'═' * 70}")
    print(f"  ZUSAMMENFASSUNG ({n} Tracks)")
    print(f"{'═' * 70}")
    print(f"  Label-Übereinstimmung: {n_match}/{n} ({100*n_match/n:.0f}%)")
    print(f"  Mittlerer Prob-Shift:  {mean_shift:.4f}  (über alle Klassen, alle Tracks)")
    print(f"  Max Prob-Shift:        {max_shift:.4f}")

    if mismatches:
        print(f"\n  Tracks mit abweichendem Label ({len(mismatches)}):")
        for r in mismatches:
            shift_per_class = "  ".join(
                f"{c}: {o:.3f}→{n:.3f} ({n-o:+.3f})"
                for c, o, n in zip(classes, r["probs_old"], r["probs_new"])
            )
            print(f"    {r['file'][:50]}")
            print(f"      {r['duration_sec']:.0f}s, {r['n_chunks']} Chunks")
            print(f"      {shift_per_class}")
    else:
        print("\n  Alle Labels identisch — kein messbarer Mismatch.")

    # Detaillierte Prob-Tabelle
    print(f"\n  Detaillierte Probabilities:")
    print(f"  {'Track':<35}  {'Chunks':>6}  ", end="")
    for c in classes:
        print(f"  {c+'_alt':>9}  {c+'_neu':>9}  {c+'_Δ':>7}", end="")
    print()
    print(f"  {'-'*35}  {'-'*6}  ", end="")
    for _ in classes:
        print(f"  {'-'*9}  {'-'*9}  {'-'*7}", end="")
    print()
    for r in results:
        print(f"  {r['file'][:35]:<35}  {r['n_chunks']:>6}  ", end="")
        for p_old, p_new in zip(r["probs_old"], r["probs_new"]):
            delta = p_new - p_old
            print(f"  {p_old:>9.4f}  {p_new:>9.4f}  {delta:>+7.4f}", end="")
        print()

    # Bewertung
    print()
    if mean_shift < 0.02:
        verdict = "MINIMAL — kein Retrain nötig, Hauptprojekt kann weitermachen."
    elif mean_shift < 0.05:
        verdict = "GERING — Modell noch nutzbar, Retrain für nächste Session empfohlen."
    elif mean_shift < 0.10:
        verdict = "MODERAT — Retrain vor dem nächsten Major-Release sinnvoll."
    else:
        verdict = "SIGNIFIKANT — Retrain dringend empfohlen."

    print(f"  Bewertung: {verdict}")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
