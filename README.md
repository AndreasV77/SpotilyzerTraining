# SpotilyzerTraining

Datenakquise, Labeling und Modell-Training für den **Spotilyzer** Hit/Mid/Flop-Klassifikator.

Dieses Repository ist das Training-Subprojekt von [Spotilyzer](https://github.com/AndreasV77/Spotilyzer). Es produziert ein trainiertes XGBoost-Modell, das anschließend ins Hauptprojekt übertragen wird.

---

## Aktueller Modellstand

Holdout-Set: 1132 Samples (20% aus 5660 validated). Stand: 2026-03-19.

| Modell | BA | Hit R. | Flop R. |
|--------|-----|--------|---------|
| `MERTv1330M_main+spotify_charts_validated_20260319` | **60.9%** | **55.1%** | 69.2% |
| `MERTv195M_main+spotify_charts_validated_20260319` | 57.4% | 47.7% | 68.7% |

**Ziele:** Flop Recall ≥ 50% ✓ — Hit Recall ≥ 80% (in Arbeit) — BA ≥ 65% (in Arbeit)

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

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install pyyaml tqdm requests pylast rapidfuzz python-dotenv
pip install mutagen
pip install pandas numpy lxml
pip install torch torchaudio transformers
pip install xgboost scikit-learn
pip install jupyter matplotlib seaborn   # optional

Copy-Item .env.example .env
# .env editieren: LASTFM_API_KEY=dein_key

New-Item -ItemType Directory -Force -Path "G:\Dev\SpotilyzerData\previews"
New-Item -ItemType Directory -Force -Path "G:\Dev\SpotilyzerData\metadata"
New-Item -ItemType Directory -Force -Path "G:\Dev\SpotilyzerData\datasets"
```

---

## Pipeline

### Haupt-Pipeline (Deezer-Scouting)

```
scout_deezer.py       →  metadata/tracks.jsonl  (Track-IDs, Ranks, Cluster)
download_previews.py  →  previews/{shard}/*.mp3  (30s-Previews, MD5-Sharding)
enrich_lastfm.py      →  tracks.jsonl  (+ playcount, listeners, tags)
compute_labels.py     →  tracks.jsonl  (+ label, robustness)
```

### Modul-Pipeline (Spotify Charts)

```
scout_spotify.py      →  datasets/spotify_charts/tracks.jsonl
download_previews.py --dataset spotify_charts  →  previews/ (shared)
```

### Modul-Pipeline (Kworb — historische Charts)

```
scout_kworb.py        →  datasets/kworb/tracks.jsonl
download_previews.py --dataset kworb  →  previews/ (shared)
```

### Gemeinsame Pipeline (ab Embeddings)

```
extract_embeddings.py [--model 95M|330M] [--dataset kworb --append]
train_model.py        [--embedder 330M] [--dataset main spotify_charts kworb] [--validated-only]
evaluate.py           [--embedder 330M] [--dataset main spotify_charts kworb] [--validated-only] [--save-report]
```

### Starten

```powershell
# Interaktives Menü
python scripts/run_pipeline.py

# Direkte Befehle:
python scripts/scout_kworb.py --dry-run
python scripts/scout_kworb.py --min-streams 20000000 --max-tracks 3000 --skip-mb
python scripts/scout_spotify.py --input-dir G:/Dev/SpotilyzerData/spotify/YYYY-MM-DD
python scripts/train_model.py --embedder 330M --dataset main spotify_charts --validated-only
python scripts/evaluate.py --embedder 330M --dataset main spotify_charts --validated-only --save-report
```

---

## Datenquellen

| Quelle | Zweck | Auth |
|--------|-------|------|
| [Deezer API](https://developers.deezer.com/api) | Audio-Previews + Popularity-Rank | keine |
| [Last.fm API](https://www.last.fm/api) | Playcount + Listeners (Validierung) | API-Key |
| [Spotify Charts](https://charts.spotify.com) | Top 200 Charts CSV (manuell) | Login |
| [Kworb.net](https://kworb.net) | Historische Chart-Daten (peak, weeks) | keine |
| [MusicBrainz API](https://musicbrainz.org/doc/MusicBrainz_API) | ISRC-Lookup (1 req/s) | keine |

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

---

## Datenstruktur

Audiodateien und Metadaten werden **außerhalb des Repos** gespeichert (`G:/Dev/SpotilyzerData/`):

```
G:/Dev/SpotilyzerData/
├── previews/
│   └── {md5[:2]}/           # MD5-Sharding (256 Ordner)
│       └── {track_id}.mp3
├── metadata/
│   └── tracks.jsonl          # Haupt-Datensatz (Deezer-Scouting)
├── datasets/
│   ├── spotify_charts/
│   │   └── tracks.jsonl      # Spotify Top 200 Charts
│   └── kworb/
│       └── tracks.jsonl      # Kworb historische Charts
└── spotify/
    └── {YYYY-MM-DD}/
        └── regional-{country}-weekly-{date}.csv
```

---

## Deployment

Nach erfolgreichem Training das Modell ins Hauptprojekt übertragen:

```powershell
Copy-Item outputs/models/spotilyzer_model_MERTv1330M_*_validated_*.joblib ..\Spotilyzer\models\
Copy-Item outputs/reports/training_report_MERTv1330M_*_validated_*.json   ..\Spotilyzer\models\
```

---

## Verwandte Repos

- [Spotilyzer](https://github.com/AndreasV77/Spotilyzer) — GUI, CLI, Analyse-Pipeline
