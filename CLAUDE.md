# CLAUDE.md — SpotilyzerTraining

Arbeitsdokument für das Modell-Training-Subprojekt von Spotilyzer.

**Erstellt:** 2026-03-07
**Zuletzt aktualisiert:** 2026-03-18 (validated-only Training, Holdout-Fix, Per-Embedder-Params, Experiment-Label)

---

## Repository-Informationen

| | Dieses Projekt | Hauptprojekt |
|---|----------------|---------------------|
| **Zweck** | Datenakquise, Labeling, Modell-Training | GUI, CLI, Analyse-Pipeline |
| **Lokal** | `G:\Dev\source\SpotilyzerTraining` | `G:\Dev\source\Spotilyzer` |
| **GitHub** | `github.com/AndreasV77/SpotilyzerTraining` | `github.com/AndreasV77/Spotilyzer` |

---

## WICHTIG: Beziehung zum Hauptprojekt

Dieses Repository ist das **Training-Subprojekt** für Spotilyzer. Es enthält alles, was mit Datenakquise, Labeling und Modell-Training zu tun hat.

### Was gehört wohin?

| Aufgabe | Repository |
|---------|------------|
| Deezer-Scouting, Preview-Download | **SpotilyzerTraining** (hier) |
| Last.fm-Enrichment | **SpotilyzerTraining** (hier) |
| Label-Berechnung, Sample-Gewichtung | **SpotilyzerTraining** (hier) |
| MERT-Embedding-Extraktion | **SpotilyzerTraining** (hier) |
| XGBoost-Training | **SpotilyzerTraining** (hier) |
| GUI, CLI, Analyse-Pipeline | **Spotilyzer** (Hauptprojekt) |
| Fertiges Modell (.joblib) | Wird von hier → Spotilyzer kopiert |

### Interface zum Hauptprojekt

**Output dieses Projekts:**
- `outputs/models/spotilyzer_model_{embedder}_{date}.joblib` — z.B. `spotilyzer_model_MERTv1330M_20260317.joblib`
- `outputs/reports/training_report_{embedder}_{date}.json` — Trainings-Metadaten

**Deployment:**
```powershell
# Nach erfolgreichem Training (Dateinamen anpassen!):
Copy-Item outputs/models/spotilyzer_model_MERTv195M_*.joblib ..\Spotilyzer\models\
Copy-Item outputs/reports/training_report_MERTv195M_*.json ..\Spotilyzer\models\
```

### Bei GUI/CLI-bezogenen Fragen

→ Siehe `G:\Dev\source\Spotilyzer\CLAUDE.md`

**NICHT in diesem Repo:**
- Analyse-Pipeline ändern
- GUI-Features entwickeln
- Export-Formate anpassen

---

## Projektziel

Verbesserung des Hit/Mid/Flop-Klassifikators für Spotilyzer.

### Aktueller Modellstand (Stand 2026-03-18)

Alle Metriken auf echtem Holdout-Set (20%, 963 Samples). Datensatz: 4813 validated-only Samples.

| Modell | Params | BA | Hit R. | Flop R. | Status |
|--------|--------|----|--------|---------|--------|
| `MERTv1330M_validated_20260318` | tuned | 55.7% | 30.4% | 72.3% | Aktiv |
| `MERTv195M_validated_20260318` | tuned | 53.8% | 27.2% | 70.4% | Trainiert |
| `MERTv195M_origparams_validated_20260318` | original | 52.6% | 24.8% | 69.2% | Referenz |

**Ursachenanalyse der früheren 26% Flop Recall:** 3900 "contested" Tracks (Deezer/Last.fm Widerspruch) wurden alle als "mid" gelabelt → Mid von 2114 auf 6032 aufgebläht (3×). Behobener Datensatz via `--validated-only`.

**Aktuelles Kernproblem:** Hit Recall 27–30% trotz balanciertem Datensatz. Ursache: nur 623 Hit-Samples (12.9%), zu ähnlich zu Mid. Flop Recall-Ziel (≥50%) erreicht.

**Parameter-Befund (95M):** Tuned vs. origparams → marginaler Unterschied (+1.2% BA). Die Regularisierungsanpassungen waren auf dem kleinen Datensatz leicht vorteilhaft, aber nicht entscheidend. Bei größerem Datensatz könnte origparams kompetitiver sein.

**Stehende Testregel:** Bei jedem neuen Datensatz BEIDE 95M-Varianten testen (`origparams` + `tuned`).

**Nächste Priorität:** Mehr Hit-Samples (Ziel ≥2000) durch Chart-Erweiterung.

---

## Datenstruktur (SpotilyzerData)

**Speicherort:** `G:/Dev/SpotilyzerData` (außerhalb des Repos, zu groß für Git)

```
G:/Dev/SpotilyzerData/
├── previews/                      # Audio-Dateien
│   ├── 00/ ... ff/                # MD5-Hash-Sharding (256 Ordner)
│   │   └── {track_id}.mp3         # Deezer-ID als Dateiname
│   └── ...
│
├── metadata/
│   └── tracks.jsonl               # Eine Zeile pro Track (alle Metadaten)
│
└── playlists/                     # Bei Bedarf generierte M3U8-Playlists
    └── *.m3u8
```

### Preview-Dateien

**Dateiname:** `{deezer_track_id}.mp3` (z.B. `3770028292.mp3`)

**WICHTIG:** Cluster-Zuordnung ist NICHT im Dateinamen! Ein Track kann mehreren Clustern angehören (z.B. ein Metal-Track der auch in den Charts ist). Cluster-Info nur in `tracks.jsonl`.

**Ordner-Sharding:** MD5-Hash der Track-ID (erste 2 Zeichen)

```python
import hashlib

def get_shard_dir(track_id: int) -> str:
    """Berechnet Shard-Verzeichnis aus Track-ID."""
    h = hashlib.md5(str(track_id).encode()).hexdigest()
    return h[:2]

def get_preview_path(track_id: int, base_path: str = "G:/Dev/SpotilyzerData/previews") -> str:
    """Vollständiger Pfad zu einer Preview-Datei."""
    shard = get_shard_dir(track_id)
    return f"{base_path}/{shard}/{track_id}.mp3"

# Beispiele:
# 3770028292 → previews/a7/3770028292.mp3
# 1234567    → previews/e1/1234567.mp3
```

**ID3-Tags (beim Download gesetzt):**
- `TIT2` — Title
- `TPE1` — Artist
- `TALB` — Album
- `COMM` — Comment: `deezer:{track_id}|clusters:{cluster1,cluster2}`

**Dependency:** `mutagen` für ID3-Tagging

### Metadaten (tracks.jsonl)

Eine JSON-Zeile pro Track. **Primärschlüssel:** `track_id` (Deezer Track-ID)

```jsonl
{"track_id": 3770028292, "title": "Song Name", "artist": "Artist Name", "album": "Album", "clusters": ["rock", "charts_us"], "deezer_rank": 895000, "lastfm_playcount": 12500000, "lastfm_listeners": 450000, "lastfm_tags": ["rock", "alternative"], "file_path": "previews/a7/3770028292.mp3", "label": "hit", "robustness": "validated"}
```

**Pflichtfelder:**
- `track_id` — Deezer Track-ID (Primärschlüssel)
- `title`, `artist`, `album` — Metadaten
- `clusters` — Liste der zugehörigen Genre-Cluster
- `deezer_rank` — Deezer Popularity-Wert
- `file_path` — Relativer Pfad zur Preview-Datei

**Optionale Felder (nach Enrichment/Labeling):**
- `lastfm_playcount`, `lastfm_listeners`, `lastfm_tags`
- `label` — hit/mid/flop (nach Label-Berechnung)
- `robustness` — validated/single_source/contested

### Playlists (M3U8)

Extended M3U Format für lesbare Tracklisten:

```m3u8
#EXTM3U
#EXTINF:30,Artist Name - Track Title
previews/a7/3770028292.mp3
#EXTINF:30,Another Artist - Another Track
previews/e1/1234567.mp3
```

**Generierung bei Bedarf** über Utility-Funktion in `scripts/utils/playlist.py`.

---

## Verzeichnisstruktur (Repository)

```
SpotilyzerTraining/
├── CLAUDE.md                    # Dieses Dokument
├── .env                         # API-Keys (LASTFM_API_KEY, nicht committen!)
├── .env.example                 # Template für .env
├── .gitignore
│
├── configs/
│   ├── clusters.yaml            # Genre-Cluster-Definitionen mit Seed-Artists
│   ├── paths.yaml               # Pfade (Preview-Speicherort etc.)
│   ├── thresholds.yaml          # Rank/Plays-Schwellenwerte für Labels
│   └── training.yaml            # Hyperparameter für XGBoost
│
├── scripts/
│   ├── run_pipeline.py          # Orchestrierungs-Skript (Haupteinstieg)
│   ├── scout_deezer.py          # Deezer-Scouting (Genre-Cluster + Charts)
│   ├── download_previews.py     # Preview-Download (mit ID3-Tagging + Sharding)
│   ├── enrich_lastfm.py         # Last.fm-Anreicherung
│   ├── compute_labels.py        # Multi-Source-Label-Berechnung
│   ├── extract_embeddings.py    # MERT-Embedding-Extraktion
│   ├── train_model.py           # XGBoost-Training mit Sample Weights
│   ├── evaluate.py              # Metriken + Confusion Matrix (Holdout-Set aus Bundle)
│   ├── inspect_dataset.py       # Read-only Diagnose-Tool (Label-Verteilung, Robustheit, etc.)
│   ├── _utils.py                # Shared helpers (logging, config-loader)
│   └── utils/
│       ├── __init__.py
│       ├── paths.py             # get_shard_dir(), get_preview_path()
│       ├── playlist.py          # create_playlist(), find_track()
│       └── metadata.py          # JSONL lesen/schreiben/updaten
│
├── data/                        # Legacy, wird nicht mehr verwendet
│   └── .gitkeep
│
├── logs/                        # Log-Dateien
│   ├── scout_YYYY-MM-DD.log
│   ├── enrichment_YYYY-MM-DD.log
│   └── training_YYYY-MM-DD.log
│
├── notebooks/                   # Jupyter für Exploration
│   └── exploration.ipynb
│
└── outputs/
    ├── models/                  # Trainierte Modelle
    │   └── spotilyzer_model_{embedder}_{date}.joblib  # z.B. MERTv1330M_20260317
    ├── reports/                 # Evaluations-Reports
    │   └── training_report_{embedder}_{date}.json
    └── embeddings/              # MERT-Embeddings (je Modell ein Unterordner)
        ├── MERT-v1-95M/         # 768-dim embeddings
        │   ├── embeddings.npy       # Embedding-Vektoren [N×768]
        │   ├── embeddings_meta.csv  # Track-Metadaten (ID, Pfad, etc.)
        │   └── embeddings_info.json # Modell, Dim, Timestamp
        └── MERT-v1-330M/        # 1024-dim embeddings
            ├── embeddings.npy
            ├── embeddings_meta.csv
            └── embeddings_info.json
```

---

## Setup

```powershell
cd G:\Dev\source\SpotilyzerTraining

# Virtuelle Umgebung
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Dependencies
pip install pyyaml tqdm requests pylast rapidfuzz python-dotenv
pip install mutagen                        # Für ID3-Tagging
pip install pandas numpy                   # Für Embeddings-Pipeline (embeddings_meta.csv)
pip install torch torchaudio transformers  # Für MERT
pip install xgboost scikit-learn           # Für Training
pip install jupyter matplotlib seaborn     # Für Notebooks (optional)

# API-Keys einrichten
Copy-Item .env.example .env
# Dann .env editieren und LASTFM_API_KEY eintragen

# Datenverzeichnis anlegen (falls noch nicht vorhanden)
New-Item -ItemType Directory -Force -Path "G:\Dev\SpotilyzerData\previews"
New-Item -ItemType Directory -Force -Path "G:\Dev\SpotilyzerData\metadata"
New-Item -ItemType Directory -Force -Path "G:\Dev\SpotilyzerData\playlists"
```

---

## Workflow: Das Orchestrierungs-Skript

**Haupteinstieg:** `python scripts/run_pipeline.py`

### Modi

```powershell
# Interaktives Menü (M = Embedder wechseln, V = validated-only toggle)
python scripts/run_pipeline.py

# Mit Flags:
python scripts/run_pipeline.py --model 95M --validated-only
python scripts/run_pipeline.py --model 330M --validated-only

# Einzelne Skripte direkt:
python scripts/extract_embeddings.py --model 95M
python scripts/train_model.py --embedder 95M --validated-only
python scripts/evaluate.py --embedder 95M --validated-only --save-report

# Explizites Modell evaluieren (bei experiment_label nötig, da Autodetect nicht greift):
python scripts/evaluate.py --model outputs/models/spotilyzer_model_MERTv195M_origparams_validated_20260318.joblib --embedder 95M --validated-only --save-report

# Datensatz-Diagnose (read-only, kein Training):
python scripts/inspect_dataset.py                    # Konsole
python scripts/inspect_dataset.py --report           # + JSON nach outputs/reports/
python scripts/inspect_dataset.py --validated-only   # Nur validated-Subset analysieren
```

**experiment_label in training.yaml:** Optionales Freitext-Label, das im Modell- und Report-Dateinamen erscheint (`experiment_label: "origparams"`). Nach Abschluss des Experiments auf `""` zurücksetzen.

**WICHTIG:** Schritte 1–4 (Scout/Download/Enrich/Labels) nur bei Datensatz-Erweiterung ausführen. Für reines Neutraining eines Modells nur Schritte 5–7 (Embeddings → Train → Evaluate).

### analyze_clusters.py — Multi-Purpose Analyse-Tool

```powershell
# Chart-Discovery: Welche weiteren Länder haben Deezer-Chart-Playlists?
# → gibt YAML-Snippet aus, das direkt in clusters.yaml kopiert werden kann
python scripts/analyze_clusters.py --discover-charts

# Sanity-Check: Sind konfigurierte Playlist-IDs noch gültig?
python scripts/analyze_clusters.py --sanity

# Cluster-Statistiken aus tracks.jsonl (Label-Verteilung, Rank-Statistiken)
python scripts/analyze_clusters.py --cluster-stats

# Track-Overlap zwischen Genre-Clustern
python scripts/analyze_clusters.py --overlap

# Vollständiger Report (alle Checks)
python scripts/analyze_clusters.py --full
python scripts/analyze_clusters.py --full --output outputs/reports/cluster_analysis.md
```

**Typischer Einsatz:** Vor jedem Scouting-Lauf `--sanity`, nach Scouting `--cluster-stats` + `--label-distribution` für Realitätscheck.

### Abhängigkeiten zwischen Schritten

```
1. scout_deezer.py
    ↓ metadata/tracks.jsonl (initial: track_id, title, artist, album, clusters, deezer_rank)
2. download_previews.py
    ↓ previews/{shard}/{track_id}.mp3 (mit ID3-Tags, MD5-Sharding)
    ↓ metadata/tracks.jsonl (file_path hinzugefügt)
3. enrich_lastfm.py
    ↓ metadata/tracks.jsonl (lastfm_* Felder hinzugefügt)
4. compute_labels.py
    ↓ metadata/tracks.jsonl (label + robustness hinzugefügt)
5. extract_embeddings.py [--model 95M|330M]
    ↓ outputs/embeddings/MERT-v1-{version}/embeddings.npy + embeddings_meta.csv + embeddings_info.json
    (Checkpoint/Resume: --resume Flag, speichert alle 500 Tracks)
6. train_model.py [--embedder 95M|330M] [--validated-only]
    ↓ outputs/models/spotilyzer_model_{tag}[_{exp_label}][_validated]_{date}.joblib
    ↓ outputs/reports/training_report_{tag}[_{exp_label}]_{date}.json
    (Sample weights: compute_sample_weight("balanced") × robustness weights)
    (test_track_ids werden im Bundle gespeichert → Holdout-Evaluation in evaluate.py)
    (Per-Embedder-Params aus training.yaml: models.MERT-v1-95M / models.MERT-v1-330M)
7. evaluate.py [--embedder 95M|330M] [--validated-only] [--save-report]
    ↓ outputs/reports/evaluation_report_{model_suffix}.json
    (Testet nur auf Holdout-Set aus Bundle — nicht auf Trainingsdaten!)
    (--model explizit angeben wenn experiment_label im Dateinamen vorhanden)
```

---

## Datenquellen

### Primär: Deezer (Audio + Rank)

- **API:** Kostenlos, keine Auth für öffentliche Endpoints
- **Audio:** 30-Sekunden-Previews (intelligent ausgewählt, repräsentativ)
- **Metrik:** `rank` (0 - ~1.000.000, höher = populärer)
- **Einschränkung:** Preview-URLs expiren nach ~15 Min (frisch holen vor Download!)

### Sekundär: Last.fm (Validierung)

- **API:** Kostenlos für nicht-kommerzielle Nutzung, API-Key erforderlich
- **Metriken:** `playcount` (absolute Plays), `listeners` (unique Listeners)
- **Vorteil:** Absolute Zahlen statt relativer Ranks
- **Matching:** Über Artist + Title (Fuzzy-Matching mit rapidfuzz)

### Verworfen

| Quelle | Grund |
|--------|-------|
| Spotify API | Audio-Features entfernt (Nov 2024), Popularity entfernt (Feb 2026) |
| Shazam | Keine öffentliche API seit 2019 |
| SoundCloud | ToS verbietet ML-Training (2025) |
| YouTube | Machbar, aber Matching-Problem zu aufwändig |

---

## Genre-Cluster

### Aktuelle Cluster (23)

**Metal (7):** extreme_metal, gothic, heavy_metal, power_symphonic, modern_metal, metalcore, crossover

**Rock (5):** hard_rock, mainstream_rock, modern_rock, classic_southern_rock, alternative_rock

**Punk/Hardcore (2):** punk, hardcore

**Electronic (2):** trance, house

**Pop (2):** pop_mainstream, pop_dance

**Hip-Hop (1):** hiphop_mainstream

**R&B / Soul (1):** rnb_soul

**Country (1):** country

**Latin (1):** latin

**Indie / Folk (1):** indie_folk

**Charts:** DE, US, UK, FR, JP, BR, ES, GLOBAL

### Scouting-Ansatz je Cluster

| Cluster-ID | Deezer Genre-ID | Radio-IDs | Scouting |
|------------|----------------|-----------|----------|
| `pop_mainstream` | 132 | — | Seed-only (Pop-Radio zu breit) |
| `pop_dance` | 113 | 30951, 42122 | Radio + Seed |
| `hiphop_mainstream` | 116 | 31021, 30991 | Radio-Hauptquelle |
| `rnb_soul` | 165+169 | 30881, 42402, 38445 | Radio + Seed |
| `country` | 84 | 42282 | Radio + Seed |
| `latin` | 197 | 30941 | Radio + Seed |
| `indie_folk` | 85+466 | 30781, 42262 | Radio + Seed |

**Nicht verwendete Genres (nach Analyse):**
- Genre 106 (Electro/Techno): Kein Pop-Bezug, würde 4. Electronic-Cluster ergeben
- Genre 152 (Rock): Deezer-Radio ist Deutschrock-Mix, Overlap mit bestehenden Clustern
- Genre 464 (Heavy Metal): Deezer-Radio = Within Temptation/Helloween, Overlap mit gothic/power_symphonic
- Genre 144 (Reggae): Zu nischig, Ranks meist 300–420K (fast alles Mid)
- `hiphop_alternative`: Kein fokussiertes Deezer-Radio vorhanden

### Radio-Scouting (Hinweis für Implementierung)

Die neuen Cluster nutzen das Feld `radios` in `clusters.yaml`. Das erfordert eine Erweiterung von `scout_deezer.py` um Radio-Scouting via `/radio/{id}/tracks`.

### Cluster-Konfiguration

Cluster-Definitionen mit Seed-Artists und Radio-IDs in `configs/clusters.yaml`.

---

## Label-Strategie: Multi-Source mit Konsens

### Einzelsignale

**Deezer:**
- Hit: rank > 700.000
- Flop: rank < 300.000
- Mid: dazwischen

**Last.fm:**
- Hit: playcount > 1M AND listeners > 100k
- Flop: playcount < 100k OR listeners < 10k
- Mid: dazwischen

### Konsens-Label

| Deezer | Last.fm | → Label | Robustheit |
|--------|---------|---------|------------|
| Hit | Hit | Hit | validated |
| Hit | Flop | Mid | contested |
| Hit | — | Hit | single_source |
| Flop | Flop | Flop | validated |
| ... | ... | ... | ... |

### Sample-Gewichtung im Training

| Robustheit | Gewicht | Bedeutung |
|------------|---------|-----------|
| validated | 1.0 | Beide Quellen einig → volles Gewicht |
| single_source | 0.5 | Nur Deezer → halbes Gewicht |
| contested | 0.7 | Widerspruch → reduziert |

---

## UI-Integration (für Hauptprojekt)

### Composite Score

Das Modell liefert `hit_probability`. Für die UI wird ein gewichteter Score berechnet:

```python
composite_score = hit_probability * robustness_factor

# robustness_factor:
#   validated: 1.0
#   single_source: 0.85
#   contested: 0.7
```

### Farbbalken-Sättigung

| Robustheit | Farbsättigung |
|------------|---------------|
| validated | 100% (kräftig) |
| single_source | ~65% (heller) |
| contested | ~40% (blass) |

---

## Logging

Alle Skripte schreiben Logs nach `logs/`:

```
logs/
├── scout_2026-03-08.log        # Deezer-Scouting
├── download_2026-03-08.log     # Preview-Download
├── enrichment_2026-03-08.log   # Last.fm (inkl. Match-Fehler!)
├── labels_2026-03-08.log       # Label-Berechnung
├── training_2026-03-08.log     # Modell-Training
└── pipeline_2026-03-08.log     # Orchestrierung
```

**Wichtig für Enrichment-Log:**
- Jeder nicht gefundene Track wird geloggt
- Match-Confidence unter Schwellenwert wird geloggt
- API-Fehler werden mit Retry-Count geloggt

---

## Konfigurationsdateien

### configs/paths.yaml

```yaml
paths:
  # Externes Datenverzeichnis (NICHT im Repo)
  data_root: "G:/Dev/SpotilyzerData"

  # Preview-Dateien (MD5-Sharding)
  previews: "G:/Dev/SpotilyzerData/previews"

  # Metadaten (JSONL)
  metadata: "G:/Dev/SpotilyzerData/metadata"

  # Playlists (M3U8)
  playlists: "G:/Dev/SpotilyzerData/playlists"

  # Embeddings (können im Repo bleiben)
  embeddings: "./outputs/embeddings"

  # Hauptprojekt (für Model-Deployment)
  main_project: "../Spotilyzer"
  main_project_models: "../Spotilyzer/models"
```

### configs/thresholds.yaml

```yaml
deezer:
  hit_threshold: 700000
  flop_threshold: 300000

lastfm:
  hit_playcount: 1000000      # 1M (gesenkt von 5M)
  hit_listeners: 100000        # 100k (gesenkt von 500k)
  flop_playcount: 100000       # 100k (gesenkt von 500k)
  flop_listeners: 10000        # 10k (gesenkt von 50k)

sample_weights:
  validated: 1.0
  single_source: 0.5
  contested: 0.7

composite_score:
  robustness_factors:
    validated: 1.0
    single_source: 0.85
    contested: 0.7
```

### configs/training.yaml

```yaml
embedder:
  model: "m-a-p/MERT-v1-95M"   # Optionen: "m-a-p/MERT-v1-95M" | "m-a-p/MERT-v1-330M"

# Optionales Experiment-Label (erscheint im Dateinamen, leer lassen wenn nicht benötigt)
experiment_label: ""   # z.B. "origparams" → spotilyzer_model_MERTv195M_origparams_validated_*.joblib

# Per-Embedder XGBoost-Parameter (train_model.py liest zuerst models.<short-name>.params)
# 95M  (768-dim):  max_depth=6, colsample=0.8 (weniger Overfitting-Gefahr)
# 330M (1024-dim): max_depth=4, colsample=0.6 (mehr Regularisierung für höhere Dim)
models:
  MERT-v1-95M:
    params:
      n_estimators: 500
      max_depth: 6
      learning_rate: 0.05
      subsample: 0.8
      colsample_bytree: 0.8
      min_child_weight: 3
      gamma: 0.1
      reg_alpha: 0.5
      reg_lambda: 2
      objective: "multi:softprob"
      num_class: 3
      eval_metric: "mlogloss"
  MERT-v1-330M:
    params:
      n_estimators: 500
      max_depth: 4
      learning_rate: 0.05
      subsample: 0.8
      colsample_bytree: 0.6
      min_child_weight: 3
      gamma: 0.1
      reg_alpha: 0.5
      reg_lambda: 2
      objective: "multi:softprob"
      num_class: 3
      eval_metric: "mlogloss"

# Fallback wenn kein per-Embedder-Eintrag vorhanden
model:
  type: xgboost
  params: { ... }  # wie MERT-v1-95M

early_stopping_rounds: 30
random_state: 42

target_metrics:
  flop_recall_min: 0.50
  hit_recall_min: 0.80
  balanced_accuracy_min: 0.65
```

---

## Ziel-Metriken

Alle Werte auf echtem Holdout-Set (20%). Datensatz: 4813 validated, 963 Holdout.

| Metrik | 95M_orig | 95M_tuned | 330M_tuned | Ziel |
|--------|----------|-----------|------------|------|
| Flop Recall | 69.2% ✓ | 70.4% ✓ | **72.3%** ✓ | ≥ 50% |
| Hit Recall | 24.8% ✗ | 27.2% ✗ | **30.4%** ✗ | ≥ 80% |
| Balanced Accuracy | 52.6% ✗ | 53.8% ✗ | **55.7%** ✗ | ≥ 65% |

**Flop Recall-Ziel erreicht.** Hit Recall bleibt kritisches Problem — 623 Hits (12.9%) zu wenig, zu ähnlich zu Mid. Nächste Maßnahme: mehr Hit-Samples durch Chart-Erweiterung.

---

## Offene Aufgaben

### Kurzfristig (nächste Session)
- [x] ~~95M Embeddings extrahieren~~ ✅ (2026-03-17, 8738 Samples)
- [x] ~~95M Neutraining~~ ✅ (MERTv195M_20260317, BA=47.8% — schlechter als 330M)
- [ ] `models/MODEL_COMPARISON.md` in Spotilyzer aktualisieren (neues 95M-Modell eintragen)
- [ ] 95M-Modell nach Spotilyzer/models/ deployen (optional, wenn gewünscht)
- [ ] `compute_labels.py` Bug 3 fixen: Dissent-Logik schickt Widersprüche zu "mid" statt "contested"
- [ ] `configs/thresholds.yaml` — Last.fm-Schwellenwerte kalibrieren

### Mittelfristig
- [ ] Mehr Hit-Samples: Zusätzliche Charts scouten (Ziel: ≥2000 Hits)
  - Priorität 1: IT, MX, CA, AU (große Märkte, bestätigt via --discover-charts)
  - Priorität 2: PL, NL, SE, KR (Nischen-Stärken: Metal/EDM/K-Pop)
  - `analyze_clusters.py --discover-charts` zuerst ausführen → prüft, ob Playlist-IDs existieren
- [ ] Ersten vollständigen Scouting-Lauf starten (alle 23 Cluster auf neuem Datensatz)
- [ ] Genre-balanced Sampling evaluieren
- [ ] LightGBM als Alternative testen
- [ ] Bestehende CSV-Daten in `scout_results/` und `scout_results_deezer/` → JSONL migrieren (einmalig, optional)

### Erledigt
- [x] JSONL-Refactoring (statt CSV/pandas)
- [x] MD5-Sharding für Previews
- [x] 7 neue Genre-Cluster (23 gesamt)
- [x] Radio-Scouting in `scout_deezer.py`
- [x] `scripts/utils/` mit `paths.py`, `playlist.py`, `metadata.py`
- [x] Label-swap-bug fix (alphabetical LabelEncoder → target_names korrekt)
- [x] compute_sample_weight("balanced") × robustness weights
- [x] Embedding-Checkpoint/Resume-System (--resume, alle 500 Tracks)
- [x] Modell-Auswahl in run_pipeline.py (interaktives Menü + --model CLI-Flag)
- [x] `--embedder`-Flag in train_model.py und evaluate.py
- [x] Embedder-Unterordner in outputs/embeddings/ (MERT-v1-95M/ vs MERT-v1-330M/)
- [x] Modell-Naming-Schema: spotilyzer_model_{embedder}_{date}.joblib
- [x] 8738-Sample-Datensatz (DE, US, UK, FR, BR, ES Charts + Genre-Cluster)
- [x] 330M-Modell trainiert und evaluiert (MERTv1330M_20260317)
- [x] MODEL_COMPARISON.md Cheat-Sheet erstellt

### Langfristig
- [ ] YouTube Views als dritte Quelle
- [ ] Genre-spezifische Modelle
- [ ] Test auf KI-generierten Tracks (Mureka, Suno)

### Backlog / Irgendwann genauer beleuchten
- [ ] **QQ Music / NetEase Cloud Music (China)** — Detaillierte R1-Recherche März 2026:
  
  | Aspekt | QQ Music | NetEase |
  |--------|----------|----------|
  | Nutzer | ~800M | ~180M |
  | Stärke | Mainstream-Pop, C-Pop | Indie, K-Pop, Electronic |
  | West-Katalog | Majors ~100%, Indie dünn | Majors ~85-90% (UMG-Deal Jan 2026) |
  | Metriken | `commentnum`, `favnum`, `listennum` | Comment Count, Popularity-Score (0-100) |
  | Python | `qqmusic-api-python` | `pyncm` oder Node.js-Proxy (Binaryify) |
  | Blocking | China Residential Proxy zwingend | Proxy + Cookie-Rotation |
  
  **Mehrwert vs. Deezer:** Comment Count / Engagement-Metriken (Frühindikator für Viralität in China)
  
  **Realistische Einschätzung:**
  - Mainstream-Pop/Hip-Hop → QQ Music ideal
  - Indie/K-Pop/Electronic → NetEase besser
  - Metal/Gothic/Hardcore → beide dünn ("Long-Tail-Lücke")
  - Ohne China-Infrastruktur unrealistisch
  
  **Alternative:** Chartmetric, Soundcharts, Viberate (kostenpflichtig, aber saubere APIs)
  
  **Fazit:** Interessant für Phase 2/3 bei Mainstream-Cluster-Erweiterung. Für Nischen-Genres kein Hebel.

---

## Hardware

**Aktuell:** GTX 1660 Ti (6 GB VRAM)
**Geplant:** Upgrade auf 16+ GB

**Relevanz für Training:**
- MERT-Embedding-Berechnung: ~2 GB VRAM
- XGBoost/LightGBM Training: CPU-basiert, VRAM irrelevant
- UMAP-Visualisierung: CPU, bei großen Datensätzen RAM-intensiv

---

## Referenzen

### Projekt-Dokumente (Hauptprojekt)
- `Spotilyzer/CLAUDE.md` — Hauptprojekt-Dokumentation
- `Spotilyzer/!BU/Spotilyzer_GenAI_Encoder_Analysis.md` — CLAP/HeartCLAP-Analyse
- `Spotilyzer/!BU/UVR_Index_for_Spotilyzer.md` — Stem-Separation-Optionen

### Externe
- [Last.fm API Docs](https://www.last.fm/api)
- [pylast (Python Last.fm Client)](https://github.com/pylast/pylast)
- [Deezer API Docs](https://developers.deezer.com/api)
- [XGBoost sample_weight](https://xgboost.readthedocs.io/en/stable/python/python_api.html)
- [mutagen (ID3-Tagging)](https://mutagen.readthedocs.io/)
