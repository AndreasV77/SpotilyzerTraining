# CLAUDE.md — SpotilyzerTraining

Arbeitsdokument für das Modell-Training-Subprojekt von Spotilyzer.

**Erstellt:** 2026-03-07
**Zuletzt aktualisiert:** 2026-03-08 (JSONL-Refactoring + 7 neue Genre-Cluster)

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
- `outputs/models/spotilyzer_model.joblib` — trainiertes XGBoost-Modell
- `outputs/reports/training_report.json` — Trainings-Metadaten

**Deployment:**
```powershell
# Nach erfolgreichem Training:
Copy-Item outputs/models/spotilyzer_model.joblib ..\Spotilyzer\models\
Copy-Item outputs/reports/training_report.json ..\Spotilyzer\models\
```

### Bei GUI/CLI-bezogenen Fragen

→ Siehe `G:\Dev\source\Spotilyzer\CLAUDE.md`

**NICHT in diesem Repo:**
- Analyse-Pipeline ändern
- GUI-Features entwickeln
- Export-Formate anpassen

---

## Projektziel

Verbesserung des Hit/Mid/Flop-Klassifikators für Spotilyzer. Das Hauptproblem: **26% Flop Recall** — das Modell erkennt Hits gut (93%), übersieht aber die meisten Flops.

**Ziel:** Flop Recall ≥ 50% bei Hit Recall ≥ 80%

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
│   ├── evaluate.py              # Metriken + Confusion Matrix
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
    │   └── spotilyzer_model.joblib
    ├── reports/                 # Evaluations-Reports
    │   └── training_report.json
    └── embeddings/              # MERT-Embeddings
        ├── embeddings.npy
        └── embeddings_meta.csv
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
# Interaktives Menü
python scripts/run_pipeline.py

# Oder direkt mit Flags:
python scripts/run_pipeline.py --full          # Vollständig: Scout → Download → Enrich → Train
python scripts/run_pipeline.py --scout         # Nur Deezer-Scouting
python scripts/run_pipeline.py --download      # Nur Preview-Download
python scripts/run_pipeline.py --enrich        # Nur Last.fm-Enrichment
python scripts/run_pipeline.py --train         # Nur Training (Labels + XGBoost)
```

### Abhängigkeiten zwischen Schritten

```
scout_deezer.py
    ↓ metadata/tracks.jsonl (initial: track_id, title, artist, album, clusters, deezer_rank)
download_previews.py
    ↓ previews/{shard}/{track_id}.mp3 (mit ID3-Tags)
    ↓ metadata/tracks.jsonl (file_path hinzugefügt)
enrich_lastfm.py
    ↓ metadata/tracks.jsonl (lastfm_* Felder hinzugefügt)
compute_labels.py
    ↓ metadata/tracks.jsonl (label + robustness hinzugefügt)
extract_embeddings.py
    ↓ outputs/embeddings/embeddings.npy + embeddings_meta.csv
train_model.py
    ↓ outputs/models/spotilyzer_model.joblib + training_report.json
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
- Hit: playcount > 5M AND listeners > 500k
- Flop: playcount < 500k OR listeners < 50k
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
  hit_playcount: 5000000
  hit_listeners: 500000
  flop_playcount: 500000
  flop_listeners: 50000

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
model:
  type: xgboost
  params:
    n_estimators: 200
    max_depth: 6
    learning_rate: 0.1
    subsample: 0.8
    colsample_bytree: 0.8

validation:
  strategy: stratified_kfold
  n_splits: 5

target_metrics:
  flop_recall_min: 0.50
  hit_recall_min: 0.80
  balanced_accuracy_min: 0.65

random_state: 42
```

---

## Ziel-Metriken

| Metrik | Aktuell | Ziel |
|--------|---------|------|
| Flop Recall | 26.8% | ≥ 50% |
| Hit Recall | 93.6% | ≥ 80% |
| Balanced Accuracy | 62.5% | ≥ 65% |

---

## Offene Aufgaben

### Kurzfristig
- [x] `scripts/utils/` Ordner erstellen mit `paths.py`, `playlist.py`, `metadata.py`
- [x] `scripts/_utils.py` erweitern (load_clusters_config, get_genre_clusters etc.)
- [x] `scripts/download_previews.py` anpassen (MD5-Sharding, ID3-Tags, JSONL-Update)
- [x] `scripts/scout_deezer.py` anpassen (JSONL statt CSV, kein pandas)
- [x] `scripts/enrich_lastfm.py` anpassen (JSONL statt CSV, kein pandas)
- [x] `scripts/compute_labels.py` anpassen (JSONL statt CSV, kein pandas)
- [x] `scripts/extract_embeddings.py` anpassen (Dateipfade aus JSONL statt glob)
- [x] `scripts/train_model.py` anpassen (Labels aus JSONL statt CSV)
- [x] `scripts/evaluate.py` anpassen (Labels aus JSONL statt CSV)
- [x] `scripts/run_pipeline.py` anpassen (keine CSV-Pfad-Args mehr)
- [x] `configs/paths.yaml` aktualisieren (metadata, playlists Einträge)
- [x] `configs/clusters.yaml` erstellen (Seed-Artists auslagern)
- [ ] Bestehende CSV-Daten → JSONL migrieren (einmalig)

### Mittelfristig
- [x] **Deezer-Genre-Struktur prüfen** — alle 25 Genres analysiert, Radio-Tracklists gesampelt
- [x] **Neue Genre-Cluster definiert** — 7 neue Cluster in `configs/clusters.yaml` eingetragen
- [x] `scout_deezer.py` um Radio-Scouting erweitern (`radios`-Feld aus clusters.yaml)
- [ ] Ersten vollständigen Scouting-Lauf starten (alle 23 Cluster)
- [ ] Genre-balanced Sampling evaluieren (nach erstem Lauf)
- [ ] Last.fm-Schwellenwerte kalibrieren (nach erstem Durchlauf)
- [ ] LightGBM als Alternative testen

### Langfristig
- [ ] YouTube Views als dritte Quelle
- [ ] Genre-spezifische Modelle
- [ ] Test auf KI-generierten Tracks (Mureka, Suno)

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
