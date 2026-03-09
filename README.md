# SpotilyzerTraining

Datenakquise, Labeling und Modell-Training für den **Spotilyzer** Hit/Mid/Flop-Klassifikator.

Dieses Repository ist das Training-Subprojekt von [Spotilyzer](https://github.com/AndreasV77/Spotilyzer). Es produziert ein trainiertes XGBoost-Modell, das anschließend ins Hauptprojekt übertragen wird.

---

## Projektziel

Verbesserung des Hit/Mid/Flop-Klassifikators.

| Metrik | Aktuell | Ziel |
|--------|---------|------|
| Flop Recall | 26.8% | ≥ 50% |
| Hit Recall | 93.6% | ≥ 80% |
| Balanced Accuracy | 62.5% | ≥ 65% |

---

## Voraussetzungen

- Python 3.10+
- [Last.fm API Key](https://www.last.fm/api/account/create) (kostenlos)
- GPU empfohlen für MERT-Embedding-Extraktion (~2 GB VRAM)
- Ca. 10–20 GB Speicherplatz für Audio-Previews

---

## Setup

```powershell
cd G:\Dev\source\SpotilyzerTraining

# Virtuelle Umgebung
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Dependencies
pip install pyyaml tqdm requests pylast rapidfuzz python-dotenv
pip install mutagen
pip install pandas numpy
pip install torch torchaudio transformers
pip install xgboost scikit-learn
pip install jupyter matplotlib seaborn   # optional

# API-Key eintragen
Copy-Item .env.example .env
# .env editieren: LASTFM_API_KEY=dein_key

# Datenverzeichnis anlegen
New-Item -ItemType Directory -Force -Path "G:\Dev\SpotilyzerData\previews"
New-Item -ItemType Directory -Force -Path "G:\Dev\SpotilyzerData\metadata"
New-Item -ItemType Directory -Force -Path "G:\Dev\SpotilyzerData\playlists"
```

---

## Pipeline

```
scout_deezer.py       →  tracks.jsonl  (Track-IDs, Ranks, Cluster-Zuordnung)
download_previews.py  →  previews/**/*.mp3  (30s-Previews mit ID3-Tags)
enrich_lastfm.py      →  tracks.jsonl  (+ playcount, listeners, tags)
compute_labels.py     →  tracks.jsonl  (+ label, robustness)
extract_embeddings.py →  embeddings.npy + embeddings_meta.csv
train_model.py        →  spotilyzer_model.joblib
evaluate.py           →  evaluation_report.json
```

### Starten

```powershell
# Interaktives Menü
python scripts/run_pipeline.py

# Oder direkt:
python scripts/run_pipeline.py --full      # komplette Pipeline
python scripts/run_pipeline.py --scout     # nur Scouting
python scripts/run_pipeline.py --download  # nur Download
python scripts/run_pipeline.py --enrich    # nur Last.fm-Enrichment
python scripts/run_pipeline.py --train     # nur Training
```

### Enrichment-Hinweis

Das Last.fm-Enrichment unterstützt Resume. Bei Abbruch oder API-Fehlern (HTTP 502) einfach neu starten — bereits verarbeitete Tracks werden übersprungen.

---

## Datenquellen

| Quelle | Zweck | Auth |
|--------|-------|------|
| [Deezer API](https://developers.deezer.com/api) | Audio-Previews + Popularity-Rank | keine |
| [Last.fm API](https://www.last.fm/api) | Playcount + Listeners (Validierung) | API-Key |

---

## Genre-Cluster (23)

| Gruppe | Cluster |
|--------|---------|
| Metal (7) | extreme_metal, gothic, heavy_metal, power_symphonic, modern_metal, metalcore, crossover |
| Rock (5) | hard_rock, mainstream_rock, modern_rock, classic_southern_rock, alternative_rock |
| Punk/Hardcore (2) | punk, hardcore |
| Electronic (2) | trance, house |
| Pop (2) | pop_mainstream, pop_dance |
| Hip-Hop (1) | hiphop_mainstream |
| R&B / Soul (1) | rnb_soul |
| Country (1) | country |
| Latin (1) | latin |
| Indie / Folk (1) | indie_folk |
| Charts | DE, US, UK, FR, JP, BR, ES, GLOBAL |

Cluster-Definitionen (Seed-Artists, Radio-IDs) in [`configs/clusters.yaml`](configs/clusters.yaml).

---

## Datenstruktur

Audiodateien und Metadaten werden **außerhalb des Repos** gespeichert (`G:/Dev/SpotilyzerData/`):

```
G:/Dev/SpotilyzerData/
├── previews/
│   └── {md5[:2]}/          # MD5-Sharding (256 Ordner)
│       └── {track_id}.mp3
├── metadata/
│   └── tracks.jsonl         # Single source of truth
└── playlists/
    └── *.m3u8
```

---

## Deployment

Nach erfolgreichem Training das Modell ins Hauptprojekt übertragen:

```powershell
Copy-Item outputs/models/spotilyzer_model.joblib ..\Spotilyzer\models\
Copy-Item outputs/reports/training_report.json   ..\Spotilyzer\models\
```

---

## Verwandte Repos

- [Spotilyzer](https://github.com/AndreasV77/Spotilyzer) — GUI, CLI, Analyse-Pipeline
