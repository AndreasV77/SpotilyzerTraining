# SpotilyzerTraining

Data acquisition, labeling, and model training for the **Spotilyzer** Hit/Mid/Flop classifier.

This repository is the training sub-project of [Spotilyzer](https://github.com/AndreasV77/Spotilyzer). It produces a trained XGBoost model that is subsequently transferred to the main project.

---

## Current Model Status

Holdout set: 4545 samples (20% from ~22,722 validated). As of 2026-03-19.

| Model | BA | Hit R. | Flop R. |
|--------|-----|--------|------|
| `MERTv1330M_main+spotify_charts+kworb_validated_20260319` | **64.2%** | **82.5%** | 73.5% |
| `MERTv1330M_main+spotify_charts_validated_20260319` | 60.9% | 55.1% | 69.2% |

**Goals:** Flop Recall ≥ 50% ✓ — Hit Recall ≥ 80% ✓ (82.5%) — BA ≥ 65% (64.2%, in progress)

---

## Prerequisites

- Python 3.10+
- [Last.fm API Key](https://www.last.fm/api/account/create) (free)
- GPU recommended for MERT embedding extraction (~2 GB VRAM)
- Approximately 10–20 GB storage for audio previews

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
# Edit .env: LASTFM_API_KEY=your_key

New-Item -ItemType Directory -Force -Path "G:\Dev\SpotilyzerData\previews"
New-Item -ItemType Directory -Force -Path "G:\Dev\SpotilyzerData\metadata"
New-Item -ItemType Directory -Force -Path "G:\Dev\SpotilyzerData\datasets"
```

---

## Pipeline

### Main Pipeline (Deezer Scouting)

```
scout_deezer.py       →  metadata/tracks.jsonl  (Track IDs, Ranks, Clusters)
download_previews.py  →  previews/{shard}/*.mp3  (30s previews, MD5 sharding)
enrich_lastfm.py      →  tracks.jsonl  (+ playcount, listeners, tags)
compute_labels.py     →  tracks.jsonl  (+ label, robustness)
```

### Module Pipeline (Spotify Charts)

```
scout_spotify.py      →  datasets/spotify_charts/tracks.jsonl
download_previews.py --dataset spotify_charts  →  previews/ (shared)
```

### Module Pipeline (Kworb — Historical Charts)

```
scout_kworb.py        →  datasets/kworb/tracks.jsonl
download_previews.py --dataset kworb  →  previews/ (shared)
```

### Shared Pipeline (from Embeddings)

```
extract_embeddings.py [--model 95M|330M] [--dataset kworb --append]
train_model.py        [--embedder 330M] [--dataset main spotify_charts kworb] [--validated-only]
evaluate.py           [--embedder 330M] [--dataset main spotify_charts kworb] [--validated-only] [--save-report]
```

### Running

```powershell
# Interactive menu
python scripts/run_pipeline.py

# Direct commands:
python scripts/scout_kworb.py --dry-run
python scripts/scout_kworb.py --min-streams 20000000 --max-tracks 3000 --skip-mb
python scripts/scout_spotify.py --input-dir G:/Dev/SpotilyzerData/spotify/YYYY-MM-DD
python scripts/train_model.py --embedder 330M --dataset main spotify_charts --validated-only
python scripts/evaluate.py --embedder 330M --dataset main spotify_charts --validated-only --save-report
```

---

## Data Sources

| Source | Purpose | Auth |
|--------|---------|------|
| [Deezer API](https://developers.deezer.com/api) | Audio previews + popularity rank | none |
| [Last.fm API](https://www.last.fm/api) | Playcount + Listeners (validation) | API key |
| [Spotify Charts](https://charts.spotify.com) | Top 200 Charts CSV (manual) | Login |
| [Kworb.net](https://kworb.net) | Historical chart data (peak, weeks) | none |
| [MusicBrainz API](https://musicbrainz.org/doc/MusicBrainz_API) | ISRC lookup (1 req/s) | none |

---

## Genre Clusters (23)

| Group | Clusters |
|--------|----------|
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

## Data Structure

Audio files and metadata are stored **outside the repo** (`G:/Dev/SpotilyzerData/`):

```
G:/Dev/SpotilyzerData/
├── previews/
│   └── {md5[:2]}/           # MD5 sharding (256 folders)
│       └── {track_id}.mp3
├── metadata/
│   └── tracks.jsonl          # Main dataset (Deezer scouting)
├── datasets/
│   ├── spotify_charts/
│   │   └── tracks.jsonl      # Spotify Top 200 Charts
│   └── kworb/
│       └── tracks.jsonl      # Kworb historical charts
└── spotify/
    └── {YYYY-MM-DD}/
        └── regional-{country}-weekly-{date}.csv
```

---

## Deployment

After successful training, transfer the model to the main project:

```powershell
Copy-Item outputs/models/spotilyzer_model_MERTv1330M_*_validated_*.joblib ..\Spotilyzer\models\
Copy-Item outputs/reports/training_report_MERTv1330M_*_validated_*.json   ..\Spotilyzer\models\
```

---

## Related Repos

- [Spotilyzer](https://github.com/AndreasV77/Spotilyzer) — GUI, CLI, analysis pipeline
