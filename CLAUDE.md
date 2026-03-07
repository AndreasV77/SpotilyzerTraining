# CLAUDE.md — SpotilyzerTraining

Arbeitsdokument für das Modell-Training-Subprojekt von Spotilyzer.

**Erstellt:** 2026-03-07

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

## Verzeichnisstruktur

```
SpotilyzerTraining/
├── CLAUDE.md                    # Dieses Dokument
├── .env                         # API-Keys (LASTFM_API_KEY, nicht committen!)
├── .env.example                 # Template für .env
├── .gitignore
│
├── configs/
│   ├── thresholds.yaml          # Rank/Plays-Schwellenwerte für Labels
│   ├── training.yaml            # Hyperparameter für XGBoost
│   └── paths.yaml               # Pfade (Preview-Speicherort etc.)
│
├── scripts/
│   ├── run_pipeline.py          # Orchestrierungs-Skript (Haupteinstieg)
│   ├── scout_deezer.py          # Deezer-Scouting (Genre-Cluster + Charts)
│   ├── download_previews.py     # Preview-Download
│   ├── enrich_lastfm.py         # Last.fm-Anreicherung
│   ├── compute_labels.py        # Multi-Source-Label-Berechnung
│   ├── extract_embeddings.py    # MERT-Embedding-Extraktion
│   ├── train_model.py           # XGBoost-Training mit Sample Weights
│   └── evaluate.py              # Metriken + Confusion Matrix
│
├── data/                        # CSV-Dateien (nicht in Git)
│   ├── scouted_tracks.csv
│   ├── scouted_tracks_enriched.csv
│   └── labeled_tracks.csv
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

### Externe Daten (NICHT im Repo)

**Preview-Dateien** werden NICHT im Repository gespeichert. Konfigurierbar in `configs/paths.yaml`:

```yaml
paths:
  previews: "D:/Data/SpotilyzerPreviews"  # ~2-3 GB MP3s
  embeddings: "./outputs/embeddings"       # Kann im Repo bleiben (kleiner)
```

---

## Setup

```powershell
cd G:\Dev\source\SpotilyzerTraining

# Virtuelle Umgebung
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Dependencies
pip install pandas pyyaml tqdm requests pylast rapidfuzz python-dotenv
pip install torch torchaudio transformers  # Für MERT
pip install xgboost scikit-learn           # Für Training
pip install jupyter matplotlib seaborn     # Für Notebooks (optional)

# API-Keys einrichten
Copy-Item .env.example .env
# Dann .env editieren und LASTFM_API_KEY eintragen
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
    ↓ scouted_tracks.csv
download_previews.py
    ↓ previews/*.mp3
enrich_lastfm.py
    ↓ scouted_tracks_enriched.csv
compute_labels.py
    ↓ labeled_tracks.csv
extract_embeddings.py
    ↓ embeddings.npy + embeddings_meta.csv
train_model.py
    ↓ spotilyzer_model.joblib + training_report.json
```

---

## Datenquellen

### Primär: Deezer (Audio + Rank)

- **API:** Kostenlos, keine Auth für öffentliche Endpoints
- **Audio:** 30-Sekunden-Previews (intelligent ausgewählt, repräsentativ)
- **Metrik:** `rank` (0 - ~1.000.000, höher = populärer)
- **Einschränkung:** Preview-URLs expiren nach ~15 Min

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
├── scout_2026-03-07.log        # Deezer-Scouting
├── enrichment_2026-03-07.log   # Last.fm (inkl. Match-Fehler!)
├── training_2026-03-07.log     # Modell-Training
└── pipeline_2026-03-07.log     # Orchestrierung
```

**Wichtig für Enrichment-Log:**
- Jeder nicht gefundene Track wird geloggt
- Match-Confidence unter Schwellenwert wird geloggt
- API-Fehler werden mit Retry-Count geloggt

---

## Genre-Cluster

16 Cluster mit Seed-Artists:

**Metal (7):** extreme_metal, gothic, heavy_metal, power_symphonic, modern_metal, metalcore, crossover

**Rock (5):** hard_rock, mainstream_rock, modern_rock, classic_southern_rock, alternative_rock

**Punk/Hardcore (2):** punk, hardcore

**Electronic (2):** trance, house

**Zusätzlich:** Country-Charts (DE, US, UK, FR, JP, BR, ES, GLOBAL)

**Bekannte Lücken:** Kein Pop/R&B/Hip-Hop-Cluster — diese Genres kommen nur über Charts rein. Könnte Genre-Bias erklären.

---

## Konfigurationsdateien

### configs/paths.yaml

```yaml
paths:
  # Preview-Dateien (NICHT im Repo, zu groß)
  previews: "D:/Data/SpotilyzerPreviews"
  
  # Embeddings (können im Repo bleiben)
  embeddings: "./outputs/embeddings"
  
  # Hauptprojekt (für Model-Deployment)
  main_project: "../Spotilyzer"
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

### Kurzfristig (für Claude Code)
- [ ] `configs/paths.yaml` erstellen
- [ ] `scripts/run_pipeline.py` (Orchestrierung mit Menü)
- [ ] `scripts/scout_deezer.py` aus Hauptprojekt migrieren
- [ ] `scripts/download_previews.py` aus Hauptprojekt migrieren
- [ ] `scripts/extract_embeddings.py` aus Hauptprojekt migrieren
- [ ] `scripts/train_model.py` mit Sample Weights
- [ ] `scripts/evaluate.py` für Metriken
- [ ] Logging in alle Skripte einbauen
- [ ] `logs/` Verzeichnis-Handling

### Mittelfristig
- [ ] Genre-balanced Sampling evaluieren
- [ ] Fehlende Genre-Cluster (Pop, Hip-Hop, R&B) hinzufügen
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
