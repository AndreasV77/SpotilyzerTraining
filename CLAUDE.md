# CLAUDE.md — SpotilyzerTraining

Working document for the model training sub-project of Spotilyzer.

**Created:** 2026-03-07
**Last updated:** 2026-05-04 (Session 9: Embedding mismatch test performed — 63% label agreement, prob shift 0.10 → SIGNIFICANT. Retraining on 30s previews does not resolve the issue. Strategic decision on inference approach pending.)

**Important rule:** Always update CLAUDE.md after completed steps — never write based on ongoing or planned results. Always read metrics from reports, never estimate them.

**Note on AGENTS.md:** The file `AGENTS.md` in the repo root is an outdated duplicate of this CLAUDE.md (as of Session 5), created automatically for other AI assistants (OpenAI Codex etc.). It is not maintained and not tracked — this CLAUDE.md is the sole authoritative document.

---

## Repository Information

| | This Project | Main Project |
|---|----------------|---------------------|
| **Purpose** | Data acquisition, labeling, model training | GUI, CLI, analysis pipeline |
| **Local** | `G:\Dev\source\SpotilyzerTraining` | `G:\Dev\source\Spotilyzer` |
| **GitHub** | `github.com/AndreasV77/SpotilyzerTraining` | `github.com/AndreasV77/Spotilyzer` |

---

## IMPORTANT: Relationship to Main Project

This repository is the **training sub-project** for Spotilyzer. It contains everything related to data acquisition, labeling, and model training.

### What Goes Where?

| Task | Repository |
|---------|------------|
| Deezer scouting, preview download | **SpotilyzerTraining** (here) |
| Last.fm enrichment | **SpotilyzerTraining** (here) |
| Label calculation, sample weighting | **SpotilyzerTraining** (here) |
| MERT embedding extraction | **SpotilyzerTraining** (here) |
| XGBoost training | **SpotilyzerTraining** (here) |
| GUI, CLI, analysis pipeline | **Spotilyzer** (main project) |
| Finished model (.joblib) | Copied from here → Spotilyzer |

### Interface to Main Project

**Output of this project:**
- `outputs/models/spotilyzer_model_{embedder}_{date}.joblib` — e.g. `spotilyzer_model_MERTv1330M_20260317.joblib`
- `outputs/reports/training_report_{embedder}_{date}.json` — training metadata

**Deployment:**
```powershell
# After successful training (adjust filenames!):
Copy-Item outputs/models/spotilyzer_model_MERTv1330M_*.joblib ..\Spotilyzer\models\
Copy-Item outputs/reports/training_report_MERTv1330M_*.json ..\Spotilyzer\models\
```

### For GUI/CLI-Related Questions

→ See `G:\Dev\source\Spotilyzer\CLAUDE.md`

**NOT in this repo:**
- Changing analysis pipeline
- Developing GUI features
- Adjusting export formats

---

## Project Goal

Improving the Hit/Mid/Flop classifier for Spotilyzer.

### Current Model Status (as of 2026-03-20, source: evaluation_reports)

All metrics on real holdout set (20%). Dataset: validated-only.

| Model | Dataset | Holdout | BA | Hit R. | Flop R. | Status |
|--------|---------|---------|-----|--------|---------|--------|
| `MERTv1330M_main+spotify_charts+kworb_validated_20260319` | ~22,722 val. | 4545 | **64.2%** | **82.5%** | 73.5% | **Active** |
| (Session 5) `MERTv1330M_main+spotify_charts+kworb_validated_20260319` | ~8960 val. | 1173 | 63.0% | 72.8% | 68.7% | Superseded |
| `MERTv1330M_main+spotify_charts_validated_20260319` | 5660 val. | 1132 | 60.9% | 55.1% | 69.2% | Predecessor |
| `MERTv195M_main+spotify_charts_validated_20260319` | 5660 val. | 1132 | 57.4% | 47.7% | 68.7% | Predecessor |
| `MERTv1330M_validated_20260318` | 5262 val. | 967 | 57.5% | 37.5% | 71.1% | Predecessor |
| `MERTv195M_validated_20260318` | 5262 val. | 967 | 53.2% | 27.3% | 68.9% | Predecessor |
| `MERTv195M_origparams_validated_20260318` | 5262 val. | 967 | 52.6% | 24.8% | 69.2% | Reference |

**Session 7 Finding: Balancing Experiments (2026-03-20)** — 4 experiments on holdout set S6 (n=4545, except expA/C with reduced holdout):

| Experiment | Configuration | BA | Hit R. | Mid R. | Flop R. |
|-----------|--------------|-----|--------|--------|---------|
| Baseline S6 | Standard (max_depth=4, col=0.6) | **64.2%** | **82.5%** | 36.6% | **73.5%** |
| expA | max_hits=6000 | 62.8% | 73.1% | 41.2% | 74.2% |
| expB | boost mid×1.5, flop×1.2 | 64.0% | 69.0% | **50.3%** | 72.8% |
| expC | max_hits=6000 + boost | 62.1% | 58.5% | 57.1% | 70.8% |
| expDim | max_depth=6, colsample=0.8 | 62.3% | **90.8%** | 30.8% | 65.3% |

No experiment beats the baseline in BA. Clear findings:
- **Undersampling (expA/C):** Hit Recall −9–24pp, Mid Recall +5–21pp — unfavorable trade
- **Boost (expB):** BA stable (−0.2pp), Mid Recall +14pp, but Hit Recall −14pp
- **expDim:** Hit Recall **90.8%** (+8.3pp!), but Flop −8.2pp, Mid −6pp, BA −1.9pp
- BA ≥ 65% not yet reached — post-hoc adjustment as next strategy (Session 8)

**Session 8 Finding: depth-Sweep + Post-hoc Adjustment (2026-03-31)**

depth-Sweep (hypothesis: sweet spot at max_depth=5) — all on holdout n=4545:

| Experiment | depth | colsample | BA | Hit R. | Mid R. | Flop R. |
|-----------|-------|-----------|-----|--------|--------|---------|
| Baseline S6 | 4 | 0.6 | **64.2%** | 82.5% | 36.6% | **73.5%** |
| expD5a | 5 | 0.6 | 63.5% | 86.3% | 35.4% | 68.9% |
| expD5b | 5 | 0.8 | 63.0% | 86.9% | 34.7% | 67.5% |
| expDim (Ref.) | 6 | 0.8 | 62.3% | **90.8%** | 30.8% | 65.3% |

Sweet spot hypothesis disproved: monotone trend — more depth/colsample → Hit Recall ↑, BA/Flop/Mid ↓. **depth=4, col=0.6 remains the BA optimum.** Hyperparameter space exhausted.

Post-hoc Logit Adjustment τ-Sweep on Baseline S6 (`_20260319`, n=4545):

| τ | BA | Hit R. | Mid R. | Flop R. |
|---|-----|--------|--------|---------|
| 0.0 (Baseline) | 64.2% | **82.5%** | 36.6% | 73.5% |
| **0.25** | **65.3%** | 73.2% | 42.9% | 79.8% |
| 0.5 | 64.8% | 61.2% | 48.5% | 84.6% |

Combined best (τ=0.25, θ_hit=0.45, θ_flop=0.35): BA=**65.7%**, Hit=76.4%. BA target achievable, but BA ≥ 65% and Hit Recall ≥ 80% simultaneously not achievable via post-hoc adjustment.

Leakage finding: 88.4% of holdout tracks from artists also present in training → all metrics optimistically biased. For unbiased evaluation: GroupKFold with artist_id required.

**Session 9 Finding: Embedding Mismatch Test (2026-05-04)**

Background: The main project (Spotilyzer) was switched to full-track analysis — MERT now processes the entire track (30s chunks → mean-pool) instead of the first 30s as before. Test with 19 full-length tracks (`scripts/test_embedding_mismatch.py`):

| Metric | Result |
|--------|--------|
| Label agreement (old vs. new) | 12/19 = **63%** |
| Mean prob shift | **0.1036** |
| Max prob shift | 0.3557 |
| Assessment | **SIGNIFICANT** |

Extreme outliers: AndreasV — Alive in the Night (Euphoria Mix): flop→hit (shift 0.53), Become (Tri-Funk): flop→mid (shift 0.40). Songs with long instrumental passages (Sloe Gin, Nothing Else Matters, Hey Joe) flip to Flop — later chunks are calmer/more instrumental than the Deezer preview.

**Core problem:** Retraining on 30s previews would not change anything — 1 chunk at 30s → mean of 1 embedding = identical result. The mismatch is structural:

| | Training | Inference (main project new) |
|--|--|--|
| Input | Deezer 30s preview (curated) | Full track, all 30s chunks averaged |
| Embedding | 1 clip → 1 embedding | N chunks → mean-pool |

**Options (decision pending):**
- A) Inference back to single-clip (energy-max) → consistent with training, no mismatch
- B) Acquire full tracks for training → correct, but costly
- C) Accept and document mismatch
- D) Two scores: XGBoost with single-clip (hit potential), CLAP chunking separately (mood/genre)

**Open question training data:** ~200k songs from private library as training source? Likely not suitable (~80% Rock/Metal, mostly older titles → dataset bias).

**Session 6 Finding:** kworb expanded to 12 markets (+ fr/au/ca with weight 0.85, it/se/nl with 0.70). Bug fix: HIT_THRESHOLDS only knew weights 1.0/0.85/0.70 — new 0.5 markets would never have been classified as Hit. 15,684 new tracks, 16,481 new previews. After dedup fix (35,530 → 26,004 embeddings): training dataset 22,722 validated, 14,991 Hits. Hit Recall: 72.8% → **82.5% (+9.7pp) — Primary target ≥80% reached**. Mid Recall dropped from ~46% to 36.6% (Mid class eroded by hit flood). BA 64.2% — 0.8pp remaining to target ≥65%.

**Session 5 Finding:** kworb module (Kworb.net _weekly_totals, 6 markets) delivered 2738 new tracks, 2497 Hits → Hit count tripled from 1216 to ~3700. Hit Recall 330M: 55.1% → 72.8% (+17.7pp). Trend stable: each +600 Hits → +17–18pp Hit Recall. Kworb track IDs were already all present in embeddings (popular tracks from Deezer scouting already captured). Confusion: 137 Hits classified as Mid — Mid class remains the largest source of errors.

**Session 4 Finding:** spotify_charts module delivered 960 new tracks, 579 Hits → Hit count almost doubled from 637 to 1216. Hit Recall 330M: 37.5% → 55.1% (+17.6pp). Hypothesis confirmed: pure data problem, not a hyperparameter problem.

**Root cause analysis of earlier 26% Flop Recall:** 3900 "contested" tracks (Deezer/Last.fm contradiction) were all labeled "mid" → Mid inflated from 2114 to 6032 (3×). Fixed dataset via `--validated-only`.

**Parameter Finding (95M):** Tuned vs. origparams → marginal difference (+0.6% BA). With larger dataset origparams could be more competitive — test rule paused for now, data volume is priority.

**Strategic Consequence:** Hit Recall 72.8% — 7.2pp remaining to target ≥80%. Next step: further data growth (more markets in Kworb, new Spotify Charts snapshots) or hyperparameter tuning.

### Current Dataset Status (2026-03-19, Session 6)

Combined dataset: Main JSONL (Deezer scouting) + spotify_charts module + kworb module

| Source | Tracks | Validated | Hits (val.) | Embeddings |
|--------|--------|-----------|-------------|------------|
| main (Deezer) | 9,661 | 5,262 | 637 | 8,794 |
| spotify_charts | 960 | 960 | 579 | 960 |
| kworb | ~18,900 | ~18,900 | ~14,000 | 26,004 (dedup, incl. main overlap) |
| **Total (dedup)** | **~28,400** | **~22,722** | **~14,991** | **26,004** |

**Holdout Set (Session 6):** 4545 samples (415 Flops, 2999 Hits, 1131 Mids) — 20% from ~22,722 validated

**Spotify Charts covered (2026-03-19):**
- `regional-{us/gb/de/jp/br/mx/global}-weekly-2026-03-12.csv`
- Path: `G:/Dev/SpotilyzerData/spotify/2026-03-19/`
- Match rate: 978/994 (98.4%) via Deezer search; 16 misses (likely JP Kanji)

**Kworb covered (2026-03-19, Session 6):**
- Markets: us, gb, de, jp, br, mx (weight 1.0/0.85) + fr, au, ca (0.85) + it, se, nl (0.70) — `_weekly_totals` (cumulative history since 2013)
- Filter: Total ≥ 20,000,000 streams → 18,928 unique tracks after dedup
- Match rate: 16,524/16,841 (98.1%) via Deezer search; 317 misses
- ISRC: `--skip-mb` (all via artist+title search); `enrich_isrc.py` planned for later ISRC enrichment
- Labels: 12,998 Hits, 3,526 Mids (kworb dataset alone)

**Next step:** BA ≥ 65% (0.8pp remaining) — options: strengthen Mid class (hyperparameter tuning, compute_labels.py Bug 3), new chart sources (ODJC, aCharts).

---

## Data Structure (SpotilyzerData)

**Location:** `G:/Dev/SpotilyzerData` (outside repo, too large for Git)

```
G:/Dev/SpotilyzerData/
├── previews/                      # Audio files (shared across all datasets)
│   ├── 00/ ... ff/                # MD5 hash sharding (256 folders)
│   │   └── {track_id}.mp3         # Deezer ID as filename
│   └── ...
│
├── metadata/
│   └── tracks.jsonl               # Main dataset (Deezer scouting)
│
├── datasets/                      # Module datasets (separate JSONL per module)
│   ├── spotify_charts/
│   │   └── tracks.jsonl           # Spotify Top 200 Charts
│   └── kworb/
│       ├── tracks.jsonl           # Kworb _weekly_totals (cumulative history)
│       ├── isrc_cache.json        # MusicBrainz ISRC cache
│       └── deezer_miss_cache.json # Tracks without Deezer match (resume skip)
│
├── spotify/                       # Raw Spotify Charts CSVs (manually downloaded)
│   └── {YYYY-MM-DD}/
│       └── regional-{country}-weekly-{date}.csv
│
└── playlists/                     # M3U8 playlists generated on demand
    └── *.m3u8
```

### Preview Files

**Filename:** `{deezer_track_id}.mp3` (e.g. `3770028292.mp3`)

**IMPORTANT:** Cluster assignment is NOT in the filename! A track can belong to multiple clusters (e.g. a metal track that is also in the charts). Cluster info only in `tracks.jsonl`.

**Folder sharding:** MD5 hash of track ID (first 2 characters)

```python
import hashlib

def get_shard_dir(track_id: int) -> str:
    """Calculate shard directory from track ID."""
    h = hashlib.md5(str(track_id).encode()).hexdigest()
    return h[:2]

def get_preview_path(track_id: int, base_path: str = "G:/Dev/SpotilyzerData/previews") -> str:
    """Full path to a preview file."""
    shard = get_shard_dir(track_id)
    return f"{base_path}/{shard}/{track_id}.mp3"

# Examples:
# 3770028292 → previews/a7/3770028292.mp3
# 1234567    → previews/e1/1234567.mp3
```

**ID3 tags (set during download):**
- `TIT2` — Title
- `TPE1` — Artist
- `TALB` — Album
- `COMM` — Comment: `deezer:{track_id}|clusters:{cluster1,cluster2}`

**Dependency:** `mutagen` for ID3 tagging

### Metadata (tracks.jsonl)

One JSON line per track. **Primary key:** `track_id` (Deezer Track ID)

```jsonl
{"track_id": 3770028292, "title": "Song Name", "artist": "Artist Name", "album": "Album", "clusters": ["rock", "charts_us"], "deezer_rank": 895000, "lastfm_playcount": 12500000, "lastfm_listeners": 450000, "lastfm_tags": ["rock", "alternative"], "file_path": "previews/a7/3770028292.mp3", "label": "hit", "robustness": "validated"}
```

**Required fields:**
- `track_id` — Deezer Track ID (primary key)
- `title`, `artist`, `album` — metadata
- `clusters` — list of assigned genre clusters
- `deezer_rank` — Deezer popularity value
- `file_path` — relative path to preview file

**Optional fields (after enrichment/labeling):**
- `lastfm_playcount`, `lastfm_listeners`, `lastfm_tags`
- `label` — hit/mid/flop (after label calculation)
- `robustness` — validated/single_source/contested

### Playlists (M3U8)

Extended M3U format for readable track lists:

```m3u8
#EXTM3U
#EXTINF:30,Artist Name - Track Title
previews/a7/3770028292.mp3
#EXTINF:30,Another Artist - Another Track
previews/e1/1234567.mp3
```

**Generated on demand** via utility function in `scripts/utils/playlist.py`.

---

## Directory Structure (Repository)

```
SpotilyzerTraining/
├── CLAUDE.md                    # This document
├── .env                         # API keys (LASTFM_API_KEY, do not commit!)
├── .env.example                 # Template for .env
├── .gitignore
│
├── configs/
│   ├── clusters.yaml            # Genre cluster definitions with seed artists
│   ├── clusters_recon.yaml      # Chart categorization for recon (see chart expansion section)
│   ├── paths.yaml               # Paths (preview storage location etc.)
│   ├── thresholds.yaml          # Rank/plays thresholds for labels
│   └── training.yaml            # XGBoost hyperparameters
│
├── scripts/
│   ├── run_pipeline.py          # Orchestration script (main entry point)
│   ├── scout_deezer.py          # Deezer scouting (genre clusters + charts)
│   ├── scout_spotify.py         # Spotify Charts CSV → datasets/spotify_charts/tracks.jsonl
│   ├── download_previews.py     # Preview download (with ID3 tagging + sharding)
│   ├── enrich_lastfm.py         # Last.fm enrichment
│   ├── compute_labels.py        # Multi-source label calculation
│   ├── extract_embeddings.py    # MERT embedding extraction
│   ├── train_model.py           # XGBoost training with sample weights
│   ├── evaluate.py              # Metrics + confusion matrix (holdout set from bundle)
│   ├── inspect_dataset.py       # Read-only diagnostic tool (label distribution, robustness, etc.)
│   ├── analyze_clusters.py      # Cluster analysis: sanity check, stats, overlap, chart discovery
│   ├── recon_clusters.py        # Cluster recon: pre-check of known clusters (freshness, spam, overlap) — before scouting
│   ├── _utils.py                # Shared helpers (logging, config loader)
│   └── utils/
│       ├── __init__.py
│       ├── paths.py             # get_shard_dir(), get_preview_path()
│       ├── playlist.py          # create_playlist(), find_track()
│       └── metadata.py          # JSONL read/write/update
│
├── data/                        # Legacy, no longer used
│   └── .gitkeep
│
├── logs/                        # Log files
│   ├── scout_YYYY-MM-DD.log
│   ├── enrichment_YYYY-MM-DD.log
│   └── training_YYYY-MM-DD.log
│
├── notebooks/                   # Jupyter for exploration
│   └── exploration.ipynb
│
└── outputs/
    ├── models/                  # Trained models
    │   └── spotilyzer_model_{embedder}_{date}.joblib  # e.g. MERTv1330M_20260317
    ├── reports/                 # Evaluation reports
    │   └── training_report_{embedder}_{date}.json
    ├── recon/                   # Recon track lists (without preview URLs)
    │   └── tracks_recon_TIMESTAMP.jsonl
    └── embeddings/              # MERT embeddings (one subfolder per model)
        ├── MERT-v1-95M/         # 768-dim embeddings
        │   ├── embeddings.npy       # Embedding vectors [N×768]
        │   ├── embeddings_meta.csv  # Track metadata (ID, path, etc.)
        │   └── embeddings_info.json # Model, dim, timestamp
        └── MERT-v1-330M/        # 1024-dim embeddings
            ├── embeddings.npy
            ├── embeddings_meta.csv
            └── embeddings_info.json
```

---

## Setup

```powershell
cd G:\Dev\source\SpotilyzerTraining

# Virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Dependencies
pip install pyyaml tqdm requests pylast rapidfuzz python-dotenv
pip install mutagen                        # For ID3 tagging
pip install pandas numpy                   # For embeddings pipeline (embeddings_meta.csv)
pip install torch torchaudio transformers  # For MERT
pip install xgboost scikit-learn           # For training
pip install jupyter matplotlib seaborn     # For notebooks (optional)

# Set up API keys
Copy-Item .env.example .env
# Then edit .env and enter LASTFM_API_KEY

# Create data directory (if not yet present)
New-Item -ItemType Directory -Force -Path "G:\Dev\SpotilyzerData\previews"
New-Item -ItemType Directory -Force -Path "G:\Dev\SpotilyzerData\metadata"
New-Item -ItemType Directory -Force -Path "G:\Dev\SpotilyzerData\playlists"
```

---

## Workflow: The Orchestration Script

**Main entry point:** `python scripts/run_pipeline.py`

### Modes

```powershell
# Interactive menu (M = switch embedder, V = toggle validated-only)
python scripts/run_pipeline.py

# With flags:
python scripts/run_pipeline.py --model 95M --validated-only
python scripts/run_pipeline.py --model 330M --validated-only

# Individual scripts directly (with module dataset):
python scripts/extract_embeddings.py --model 95M --dataset spotify_charts --append
python scripts/train_model.py --embedder 95M --dataset main spotify_charts --validated-only
python scripts/evaluate.py --embedder 95M --dataset main spotify_charts --validated-only --save-report

# Without module dataset (main only):
python scripts/train_model.py --embedder 95M --validated-only
python scripts/evaluate.py --embedder 95M --validated-only --save-report

# Evaluate explicit model (when autodetect doesn't work):
python scripts/evaluate.py --model outputs/models/spotilyzer_model_MERTv195M_origparams_validated_20260318.joblib --embedder 95M --validated-only --save-report

# Scout Spotify Charts (new snapshot):
python scripts/scout_spotify.py --input-dir G:/Dev/SpotilyzerData/spotify/YYYY-MM-DD --dry-run
python scripts/scout_spotify.py --input-dir G:/Dev/SpotilyzerData/spotify/YYYY-MM-DD

# Dataset diagnostics (read-only, no training):
python scripts/inspect_dataset.py                    # Console
python scripts/inspect_dataset.py --report           # + JSON to outputs/reports/
python scripts/inspect_dataset.py --validated-only   # Analyze validated subset only
```

**experiment_label in training.yaml:** Optional free-text label that appears in the model and report filename (`experiment_label: "origparams"`). Reset to `""` after the experiment is complete.

**IMPORTANT:** Steps 1–4 (scout/download/enrich/labels) should only be run when extending the dataset. For pure model retraining, only run steps 5–7 (embeddings → train → evaluate).

### analyze_clusters.py — Multi-Purpose Analysis Tool

```powershell
# Chart discovery: which other countries have Deezer chart playlists?
# → outputs YAML snippet that can be copied directly into clusters.yaml
python scripts/analyze_clusters.py --discover-charts

# Sanity check: are configured playlist IDs still valid?
python scripts/analyze_clusters.py --sanity

# Cluster statistics from tracks.jsonl (label distribution, rank statistics)
python scripts/analyze_clusters.py --cluster-stats

# Track overlap between genre clusters
python scripts/analyze_clusters.py --overlap

# Full report (all checks)
python scripts/analyze_clusters.py --full
python scripts/analyze_clusters.py --full --output outputs/reports/cluster_analysis.md
```

**Typical use:** Reads from `clusters.yaml` (training config) + `tracks.jsonl`. For `--cluster-stats`/`--overlap`, a scouting run must have already been completed.

**Note on `--label-distribution`:** Internally runs the same code path as `--cluster-stats` — no difference in output. Status unclear (possibly planned as a standalone path but not implemented).

---

### recon_clusters.py — Cluster Pre-Check

Reconnaissance tool for known chart clusters. Reads from `configs/clusters_recon.yaml` (NOT `clusters.yaml`).

**What it does:**
- Track count, rank distribution (min/max/median/P25/P75), artist diversity
- Release dates via Album API — only for 15 sample tracks (top 5 / mid 5 / bottom 5)
- Overlap analysis between charts
- Spam detection (single-artist dominance, old releases, niche ranks)

**What it does NOT do:** Preview URLs, downloads, `tracks.jsonl` write access, Last.fm

```powershell
# Default: validated + suspicious charts
python scripts/recon_clusters.py

# Validated charts only
python scripts/recon_clusters.py --scope validated

# All including excluded (complete documentation)
python scripts/recon_clusters.py --scope all

# Specific charts (from any category)
python scripts/recon_clusters.py --charts KR AR CL

# Ad-hoc: test chart without config entry
python scripts/recon_clusters.py --add-chart VN 1234567890 "Vietnam"

# Dry-run: shows charts + estimated API calls
python scripts/recon_clusters.py --dry-run
```

**Output:**
- `outputs/reports/recon_TIMESTAMP.json` — statistics + warnings
- `outputs/recon/tracks_recon_TIMESTAMP.jsonl` — track list (without preview URLs)

**`clusters_recon.yaml` — Categories:**

| Category | Description |
|----------|-------------|
| `existing` | Already configured in `clusters.yaml` for training |
| `validated` | Official (Deezer Charts), current, no anomalies |
| `suspicious` | Potentially manipulated — requires manual decision |
| `excluded` | Not usable (outdated, user-curated, API bug) |

**Known gap:** The `existing` charts (DE, US, UK, FR, BR, ES, JP, GLOBAL) have no `playlist_id` in `clusters_recon.yaml` — they are skipped by recon even with `--scope all`. They should be supplemented with playlist IDs for a complete analysis.

**Spam detection thresholds** (from `recon_settings` in `clusters_recon.yaml`):
- Single-artist dominance > 30% → warning
- Artist diversity < 0.5 → warning
- < 30% releases from last 12 months → "Chart outdated?"
- Rank median > 900,000 → "Niche content?"

---

### Workflow: Cluster Expansion Planning

This workflow is a **prerequisite** for any new scouting run with expanded clusters. It runs before `scout_deezer.py` and is separate from the normal training pipeline.

```
1. analyze_clusters.py --discover-charts
      → finds playlist IDs for new countries via Deezer search
      → outputs YAML snippet for clusters_recon.yaml
      ↓
2. Update clusters_recon.yaml
      → enter new entries in validated/suspicious
      → (manually / by Claude in chat)
      ↓
3. recon_clusters.py
      → pre-check: freshness, spam, artist diversity, overlap
      → report in outputs/reports/recon_TIMESTAMP.json
      ↓
4. Decision: suspicious → validated or excluded
      → manually update clusters_recon.yaml
      ↓
5. Update clusters.yaml
      → enter validated clusters (if intended for scouting)
      ↓
6. analyze_clusters.py --sanity
      → checks whether all playlist_ids in clusters.yaml are reachable
      ↓
7. Cluster planning
      → which clusters for scouting? Tier assignment? Weighting?
      → basis: recon report + own assessment
      → tier system: see chart expansion section below
      ↓
8. scout_deezer.py
      → scouting exclusively for decided clusters
      ↓
9. analyze_clusters.py --cluster-stats --overlap
      → post-scouting analysis (requires tracks.jsonl)
```

---

## Chart Expansion: Status and Decisions

**As of:** 2026-03-18
**Reference documents:**
- `outputs/reports/recon_*.json` — recon reports
- `outputs/recon/tracks_recon_*.jsonl` — sample track lists
- Claude.ai Project: Consolidated index from chat logs (Obsidian workbench: `1_Continue_*.md`, `2_Datenquellen_*.md`, `3_Chat-Verlauf_*.md`)

### Background: Why Chart Expansion?

**Core problem:** Too few Hit samples (623 of 4813 = 12.9%). FR/BR/ES charts delivered almost only Mids/Flops (+10 Hits, +645 Flops, +871 Mids). Hit Recall stagnated at 27–30%.

**Goal:** ≥2000 Hit samples through systematic chart expansion.

### Discovery: Deezer Chart Infrastructure

**Finding 1: "Deezer Charts" Account**
- Account ID: **637006841** (not the editorial ID 2!)
- Semi-official account with automatically generated country charts
- All charts have ~100 tracks, updated regularly

**Finding 2: Search API Bug**
- The Deezer Search API returns **0 followers** for these playlists
- Real follower counts only via direct Playlist API call (`/playlist/{id}`)
- Example: "Top Italy" shows 0 followers in search, but 678,285 via Playlist API

**Finding 3: Release Date Limitation**
- Playlist Track API doesn't return `release_date` in the album object
- Workaround in `recon_clusters.py`: Album API call for 15 sample tracks (top 5 / mid 5 / bottom 5)

### Chart Categorization (complete)

#### Existing (8) — already in `clusters.yaml`

| Code | Name | Status |
|------|------|--------|
| DE | Germany | Active |
| US | United States | Active |
| UK | United Kingdom | Active |
| FR | France | Active |
| JP | Japan | Active |
| BR | Brazil | Active |
| ES | Spain | Active |
| GLOBAL | Worldwide | Active |

#### Validated (22) — ready for integration

**Europe:**
| Code | Playlist ID | Followers | Notes |
|------|-------------|-----------|-------|
| IT | 1116187241 | 678K | Current, Bruno Mars / Alex Warren |
| NL | 1266971851 | 273K | Current |
| SE | 1313620305 | 69K | Current |
| AT | 1313615765 | 61K | Current |
| CH | 1313617925 | 58K | Current |
| BE | 1266968331 | 152K | Current, Taylor Swift |
| NO | 1313619885 | 15K | Current |
| DK | 1313618905 | 32K | Current |
| FI | 1221034071 | 56K | Local artists! |
| IE | 1313619455 | 39K | Current |
| PL | 1266972311 | 107K | Current |

**Americas:**
| Code | Playlist ID | Followers | Notes |
|------|-------------|-----------|-------|
| MX | 1111142361 | 1.05M | Latin Hot, Peso Pluma / Bad Bunny |
| CA | 1652248171 | 42K | Current |
| CO | 1116188451 | 1.5M | Latin current, Ryan Castro |

**Asia-Pacific:**
| Code | Playlist ID | Followers | Notes |
|------|-------------|-----------|-------|
| AU | 1313616925 | 59K | Current |
| ID | 1116188761 | 338K | Current |
| PH | 1362518895 | 57K | Current |
| SG | 1313620765 | 21K | Current |
| MY | 1362515675 | 5K | Local acts |

**MENA:**
| Code | Playlist ID | Followers | Notes |
|------|-------------|-----------|-------|
| EG | 1362501615 | 111K | MENA market |
| SA | 1362521285 | 27K | MENA market |
| ZA | 1362528775 | 62K | Current |

#### Suspicious (5) — manual review required

| Code | Problem | Sample Tracks |
|------|---------|---------------|
| KR | Classical orchestra at #2/#3 — bot manipulation? | Borodine, Saint-Saëns instead of K-Pop |
| AR | BTS/Jimin only — K-Pop stan takeover | "Who", "Set Me Free", "Let Me Know" |
| CL | OLD BTS tracks only (2014!) — definitely manipulated | "Danger", "24/7=Heaven" |
| TH | Strange mix — French Star Academy at #3? | Unusual genre mix |
| PT | "Barulho Para Relaxar" = white noise tracks | Kim Wilde "You Came" (1988) |

**Decision pending:** These charts might still contain usable tracks if top positions are ignored. Requires a manual recon run with `--charts KR AR CL TH PT` and individual review.

#### Excluded (4) — not usable

| Code | Reason | Detail |
|------|--------|--------|
| TR | Outdated | "Top Turkey **2020**" — 6 years old |
| AE/UAE | User-curated | 2019, only 7 followers |
| NZ | User-curated | 293 tracks, not a real chart |
| IN | API bug | Search returns Indonesia instead of India |

#### Not searched (10) — status unclear

CN, RU, VN, TW, HK, IL, GR, CZ, HU, RO — marked as problematic in earlier sessions, reason no longer traceable. Review again if needed.

### Planned Tier System

**Concept:** Weight charts by market relevance. Not yet implemented — final assignment to follow recon data analysis (overlap, rank distribution).

| Tier | Weight | Criteria | Candidates |
|------|--------|----------|------------|
| **Tier 1** | 1.0 | International reference, defines mainstream | US, UK, GLOBAL |
| **Tier 2** | 0.85 | Large export markets, significant influence | DE, FR, AU, CA, JP, BR |
| **Tier 3** | 0.7 | Mid-sized markets, own scene | ES, IT, MX, NL, SE, KR (if validated) |
| **Tier 4** | 0.5 | Local markets, niche relevance | PL, AT, CH, BE, NO, DK, FI, IE, etc. |

**Application (planned):**
- Track in multiple charts → average of tier weights
- Track only in Tier-4 chart → `robustness * 0.5`
- Implementation in `thresholds.yaml` or `clusters.yaml` (still to be decided)

**Important:** "Weights are guesses in a suit." The tier system is a heuristic, not a scientifically validated metric. Transparency about uncertainty takes precedence over pseudo-precision.

### Spam Detection Criteria (in recon_clusters.py)

| Criterion | Threshold | Meaning |
|-----------|-----------|--------|
| Single-artist dominance | > 30% | One artist dominates the chart → streaming farm? |
| Artist diversity | < 0.5 | Few unique artists / total tracks |
| Release freshness | < 30% from last 12 months | Chart outdated? |
| Rank median | > 900,000 | Niche content instead of mainstream chart |

### Status of Deezer Chart Expansion (2026-03-19)

**Decision:** Deezer chart expansion is **not further prioritized**. Experience shows that additional Deezer country charts deliver predominantly Mids/Flops, few Hits. Instead: `kworb_deezer` module.

Remaining open points (only if needed):
1. `clusters_recon.yaml`: Add playlist IDs for DE/US/UK/FR/BR/ES/JP/GLOBAL (→ recon currently skips them)
2. KR, TH: Targeted recon run — decide after kworb_deezer implementation whether relevant
3. 22 validated charts possibly as `robustness` signal in kworb_deezer (not as primary source)

### External Chart Sources (Primary Strategy from Session 3)

Deezer charts have a hard ceiling. Primary strategy is now the `kworb_deezer` module.

**Main sources (for kworb_deezer Phase 1):**

| Source | Access | Format | Markets |
|--------|--------|--------|---------|
| **Kworb.net** | Scraping, no login | Static HTML, `pandas.read_html()` | ~70 countries, Spotify Top 200 |
| **charts.spotify.com** | Manual download, login required | CSV per country/week | ~70 countries |
| **MusicBrainz** | API (1 req/s), free | JSON | ISRC lookup for deduplication |

**Supplementary sources (Phase 2/3):**

| Source | Access | Value |
|--------|--------|-------|
| Billboard Japan | CSV download, no login | J-Pop without scraping |
| Hung Medien Network | Scraping (consistent schema) | 15 EU countries + Oceania |
| Certification DBs (BVMI, BPI, RIAA, etc.) | Publicly searchable | `robustness=validated` signal |

**Not usable:**

| Source | Reason |
|--------|--------|
| Spotify API | No stream counts; ToS prohibits scraping |
| Apple Music | No public playcount data |
| Shazam | No public API since 2019 |

---

### Step Dependencies

```
Main Pipeline (Deezer):
1. scout_deezer.py
    ↓ metadata/tracks.jsonl (initial: track_id, title, artist, album, clusters, deezer_rank)
2. download_previews.py [--dataset main]
    ↓ previews/{shard}/{track_id}.mp3 (with ID3 tags, MD5 sharding)
    ↓ metadata/tracks.jsonl (file_path added)
3. enrich_lastfm.py
    ↓ metadata/tracks.jsonl (lastfm_* fields added)
4. compute_labels.py
    ↓ metadata/tracks.jsonl (label + robustness added)

Module Pipeline (spotify_charts):
1b. scout_spotify.py --input-dir G:/Dev/SpotilyzerData/spotify/YYYY-MM-DD
    ↓ datasets/spotify_charts/tracks.jsonl (track_id, label=hit/mid, robustness=validated)
2b. download_previews.py --dataset spotify_charts
    ↓ previews/{shard}/{track_id}.mp3 (shared with main pipeline!)
    ↓ datasets/spotify_charts/tracks.jsonl (file_path added)

Module Pipeline (kworb):
1c. scout_kworb.py --min-streams 20000000 --max-tracks 3000 --skip-mb
    ↓ datasets/kworb/tracks.jsonl (chart_entries, chart_score, label, robustness=validated)
    ↓ datasets/kworb/isrc_cache.json + deezer_miss_cache.json (checkpoint system)
    (--skip-mb: skip MusicBrainz, go directly to Deezer search)
    (Checkpoint every 100 tracks: kworb_checkpoint.jsonl, deleted on completion)
2c. download_previews.py --dataset kworb
    ↓ previews/{shard}/{track_id}.mp3 (shared, usually already present!)
    ↓ datasets/kworb/tracks.jsonl (file_path added)

Shared Pipeline (from Embeddings):
5. extract_embeddings.py [--model 95M|330M] [--dataset spotify_charts --append]
    ↓ outputs/embeddings/MERT-v1-{version}/embeddings.npy + embeddings_meta.csv + embeddings_info.json
    (Checkpoint/resume: --resume flag, saves every 500 tracks)
    (--append: add new tracks to existing .npy)
6. train_model.py [--embedder 95M|330M] [--dataset main spotify_charts] [--validated-only]
    ↓ outputs/models/spotilyzer_model_{tag}[_{exp_label}][_{datasets}][_validated]_{date}.joblib
    ↓ outputs/reports/training_report_{tag}_{datasets}_{date}.json
    (Sample weights: compute_sample_weight("balanced") × robustness weights)
    (test_track_ids stored in bundle → holdout evaluation in evaluate.py)
    (Per-embedder params from training.yaml: models.MERT-v1-95M / models.MERT-v1-330M)
7. evaluate.py [--embedder 95M|330M] [--dataset main spotify_charts] [--validated-only] [--save-report]
    ↓ outputs/reports/evaluation_report_{model_suffix}.json
    (Tests only on holdout set from bundle — not on training data!)
    (Autodetect selects newest *validated*.joblib for the embedder)
```

---

## Data Sources

### Primary: Deezer (Audio + Rank)

- **API:** Free, no auth for public endpoints
- **Audio:** 30-second previews (intelligently selected, representative)
- **Metric:** `rank` (0 - ~1,000,000, higher = more popular)
- **Limitation:** Preview URLs expire after ~15 min (fetch fresh before downloading!)

### Secondary: Last.fm (Validation)

- **API:** Free for non-commercial use, API key required
- **Metrics:** `playcount` (absolute plays), `listeners` (unique listeners)
- **Advantage:** Absolute numbers instead of relative ranks
- **Matching:** Via artist + title (fuzzy matching with rapidfuzz)

### Discarded

| Source | Reason |
|--------|--------|
| Spotify API | Audio features removed (Nov 2024), Popularity removed (Feb 2026) |
| Shazam | No public API since 2019 |
| SoundCloud | ToS prohibits ML training (2025) |
| YouTube | Feasible but matching problem too costly |

---

## Genre Clusters

### Current Clusters (23)

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

### Scouting Approach per Cluster

| Cluster ID | Deezer Genre ID | Radio IDs | Scouting |
|------------|----------------|-----------|----------|
| `pop_mainstream` | 132 | — | Seed only (Pop Radio too broad) |
| `pop_dance` | 113 | 30951, 42122 | Radio + seed |
| `hiphop_mainstream` | 116 | 31021, 30991 | Radio primary source |
| `rnb_soul` | 165+169 | 30881, 42402, 38445 | Radio + seed |
| `country` | 84 | 42282 | Radio + seed |
| `latin` | 197 | 30941 | Radio + seed |
| `indie_folk` | 85+466 | 30781, 42262 | Radio + seed |

**Unused genres (after analysis):**
- Genre 106 (Electro/Techno): No pop connection, would create a 4th electronic cluster
- Genre 152 (Rock): Deezer Radio is a German rock mix, overlap with existing clusters
- Genre 464 (Heavy Metal): Deezer Radio = Within Temptation/Helloween, overlap with gothic/power_symphonic
- Genre 144 (Reggae): Too niche, ranks mostly 300–420K (almost all Mid)
- `hiphop_alternative`: No focused Deezer Radio available

### Radio Scouting (Implementation Note)

The new clusters use the `radios` field in `clusters.yaml`. This requires extending `scout_deezer.py` to support radio scouting via `/radio/{id}/tracks`.

### Cluster Configuration

Cluster definitions with seed artists and radio IDs in `configs/clusters.yaml`.

---

## Label Strategy: Multi-Source with Consensus

### Individual Signals

**Deezer:**
- Hit: rank > 700,000
- Flop: rank < 300,000
- Mid: in between

**Last.fm:**
- Hit: playcount > 1M AND listeners > 100k
- Flop: playcount < 100k OR listeners < 10k
- Mid: in between

### Consensus Label

| Deezer | Last.fm | → Label | Robustness |
|--------|---------|---------|------------|
| Hit | Hit | Hit | validated |
| Hit | Flop | Mid | contested |
| Hit | — | Hit | single_source |
| Flop | Flop | Flop | validated |
| ... | ... | ... | ... |

### Sample Weighting in Training

| Robustness | Weight | Meaning |
|------------|--------|---------|
| validated | 1.0 | Both sources agree → full weight |
| single_source | 0.5 | Deezer only → half weight |
| contested | 0.7 | Contradiction → reduced |

---

## UI Integration (for Main Project)

### Composite Score

The model returns `hit_probability`. For the UI a weighted score is calculated:

```python
composite_score = hit_probability * robustness_factor

# robustness_factor:
#   validated: 1.0
#   single_source: 0.85
#   contested: 0.7
```

### Color Bar Saturation

| Robustness | Color saturation |
|------------|------------------|
| validated | 100% (vivid) |
| single_source | ~65% (lighter) |
| contested | ~40% (pale) |

---

## Logging

All scripts write logs to `logs/`:

```
logs/
├── scout_2026-03-08.log        # Deezer scouting
├── download_2026-03-08.log     # Preview download
├── enrichment_2026-03-08.log   # Last.fm (including match errors!)
├── labels_2026-03-08.log       # Label calculation
├── training_2026-03-08.log     # Model training
└── pipeline_2026-03-08.log     # Orchestration
```

**Important for enrichment log:**
- Every track not found is logged
- Match confidence below threshold is logged
- API errors are logged with retry count

---

## Configuration Files

### configs/paths.yaml

```yaml
paths:
  # External data directory (NOT in repo)
  data_root: "G:/Dev/SpotilyzerData"

  # Preview files (MD5 sharding)
  previews: "G:/Dev/SpotilyzerData/previews"

  # Metadata (JSONL)
  metadata: "G:/Dev/SpotilyzerData/metadata"

  # Playlists (M3U8)
  playlists: "G:/Dev/SpotilyzerData/playlists"

  # Embeddings (can stay in repo)
  embeddings: "./outputs/embeddings"

  # Main project (for model deployment)
  main_project: "../Spotilyzer"
  main_project_models: "../Spotilyzer/models"
```

### configs/thresholds.yaml

```yaml
deezer:
  hit_threshold: 700000
  flop_threshold: 300000

lastfm:
  hit_playcount: 1000000      # 1M (lowered from 5M)
  hit_listeners: 100000        # 100k (lowered from 500k)
  flop_playcount: 100000       # 100k (lowered from 500k)
  flop_listeners: 10000        # 10k (lowered from 50k)

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
  model: "m-a-p/MERT-v1-95M"   # Options: "m-a-p/MERT-v1-95M" | "m-a-p/MERT-v1-330M"

# Optional experiment label (appears in filename, leave blank if not needed)
experiment_label: ""   # e.g. "origparams" → spotilyzer_model_MERTv195M_origparams_validated_*.joblib

# Per-embedder XGBoost parameters (train_model.py reads models.<short-name>.params first)
# 95M  (768-dim):  max_depth=6, colsample=0.8 (less risk of overfitting)
# 330M (1024-dim): max_depth=4, colsample=0.6 (more regularization for higher dim)
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

# Fallback when no per-embedder entry exists
model:
  type: xgboost
  params: { ... }  # same as MERT-v1-95M

early_stopping_rounds: 30
random_state: 42

target_metrics:
  flop_recall_min: 0.50
  hit_recall_min: 0.80
  balanced_accuracy_min: 0.65
```

---

## Target Metrics

All values on real holdout set (20%). Source: `evaluation_report_*.json`

### Session 5 — main + spotify_charts + kworb (~8960 validated, 1173 holdout)

| Metric | 330M | Target |
|--------|------|--------|
| Flop Recall | **68.7%** ✓ | ≥ 50% |
| Hit Recall | **72.8%** ✗ | ≥ 80% |
| Balanced Accuracy | **63.0%** ✗ | ≥ 65% |

### Session 4 — main + spotify_charts (5660 validated, 1132 holdout) — Reference

| Metric | 95M | 330M | Target |
|--------|-----|------|--------|
| Flop Recall | 68.7% ✓ | 69.2% ✓ | ≥ 50% |
| Hit Recall | 47.7% ✗ | 55.1% ✗ | ≥ 80% |
| Balanced Accuracy | 57.4% ✗ | 60.9% ✗ | ≥ 65% |

### Session 3 — main only (5262 validated, 967 holdout) — Reference

| Metric | 95M_orig | 95M_tuned | 330M_tuned |
|--------|----------|-----------|------------|
| Flop Recall | 69.2% | 68.9% | 71.1% |
| Hit Recall | 24.8% | 27.3% | 37.5% |
| Balanced Accuracy | 52.6% | 53.2% | 57.5% |

**Flop Recall target reached.** Hit Recall: each +~2500 Hits → +17–18pp. Trend stable over 3 sessions. Last step to ≥80%: further data growth or hyperparameter tuning.

---

## Open Tasks

### Short-term (next session)
- [ ] **Strategic decision on inference approach:** Option A (single-clip energy-max), B (full tracks), C (accept mismatch) or D (two scores) — see Session 9 finding
- [ ] Check private library (~200k songs) for suitability as training data (estimated: not suitable, 80% Rock/Metal, mostly old)


- [x] ~~Extract 95M embeddings~~ ✅ (2026-03-17, 8738 samples)
- [x] ~~95M retraining~~ ✅ (MERTv195M_20260317, BA=47.8% — worse than 330M)
- [x] ~~Recon run~~ ✅ (2026-03-18, all validated + suspicious charts checked)
- [x] ~~Suspicious decisions~~ ✅ AR/CL/PT → excluded; KR/TH remain suspicious
- [x] ~~Scouting run~~ ✅ (2026-03-18, --min-rank 600000, existing clusters)
- [x] ~~Embeddings --append~~ ✅ (2026-03-18, 56 new tracks, both models)
- [x] ~~Training on new dataset~~ ✅ (2026-03-18, all three models, --validated-only)
- [x] ~~spotify_charts module~~ ✅ (2026-03-19, scout_spotify.py + --dataset flag in all scripts)
- [x] ~~Training + eval on main+spotify_charts~~ ✅ (2026-03-19, 330M: BA=60.9%, Hit R.=55.1%)
- [x] ~~evaluate.py --dataset flag + autodetect fix~~ ✅ (2026-03-19)
- [x] ~~Update `models/MODEL_COMPARISON.md` in Spotilyzer~~ ✅ (2026-03-19, Session 6)
- [ ] Fix `compute_labels.py` Bug 3: Dissent logic sends contradictions to "mid" instead of "contested"
- [x] ~~Implement kworb module~~ ✅ (scout_kworb.py + checkpoint system, 2026-03-19)
- [x] ~~Training + eval on main+spotify_charts+kworb~~ ✅ (330M: BA=63.0%, Hit R.=72.8%, 2026-03-19)
- [x] ~~Expand kworb to 12 markets~~ ✅ (fr/au/ca/it/se/nl, bug fix HIT_THRESHOLDS, 2026-03-19)
- [x] ~~Reach Hit Recall ≥80%~~ ✅ (82.5%, Session 6, 2026-03-19)
- [x] ~~Balancing experiments (expA/B/C/Dim)~~ ✅ (2026-03-20) — no experiment beats baseline; expDim finding: sweet spot max_depth=5 (between 4 and 6)

### Module System: Kworb Scraper (completed)

**Status:** ✅ scout_kworb.py implemented and successfully run (2026-03-19).

**Result:** 2738 tracks, 2497 Hits, all embeddings already present. Training delivered BA=63.0%, Hit R.=72.8%.

**Open todos (nice-to-have):**
- [ ] `enrich_isrc.py` — background script: fill ISRC for `isrc: null` tracks via MusicBrainz (currently using `--skip-mb`)
- [ ] `configs/datasets/kworb.yaml` — market list, tier weights, hit thresholds (currently hardcoded in scout_kworb.py)

### Cluster Expansion Planning (Deezer — low priority)
- [x] ~~Run `recon_clusters.py`~~ ✅ (2026-03-18)
- [x] ~~AR, CL, PT~~ ✅ → excluded (manipulated/spam)
- [ ] KR, TH: targeted recon run (`--charts KR TH`) → then decide (after kworb_deezer)
- [ ] `clusters_recon.yaml`: Add `playlist_id` for DE/US/UK/FR/BR/ES/JP/GLOBAL
- [ ] Finalize tier assignment based on overlap/rank data

### Medium-term
- [x] ~~More Hit samples: target ≥2000 validated Hits~~ ✅ (~3700 Hits, Session 5)
- [ ] Evaluate genre-balanced sampling
- [ ] Test LightGBM as alternative
- [ ] `configs/thresholds.yaml` — calibrate Last.fm thresholds (or made obsolete via module system)
- [ ] Migrate existing CSV data in `scout_results/` → JSONL (one-time, optional)

### Done
- [x] spotify_charts module: scout_spotify.py, download_previews.py --dataset, extract_embeddings.py --dataset, train_model.py --dataset, evaluate.py --dataset
- [x] evaluate.py autodetect fix: Glob *validated* instead of _validated_ (matches main+spotify_charts)
- [x] JSONL refactoring (instead of CSV/pandas)
- [x] MD5 sharding for previews
- [x] 7 new genre clusters (23 total)
- [x] Radio scouting in `scout_deezer.py`
- [x] `scripts/utils/` with `paths.py`, `playlist.py`, `metadata.py`
- [x] Label swap bug fix (alphabetical LabelEncoder → target_names correct)
- [x] compute_sample_weight("balanced") × robustness weights
- [x] Embedding checkpoint/resume system (--resume, every 500 tracks)
- [x] `--append` flag in `extract_embeddings.py` (embed only new tracks, skip existing)
- [x] Model selection in run_pipeline.py (interactive menu + --model CLI flag)
- [x] `--embedder` flag in train_model.py and evaluate.py
- [x] Embedder subdirectories in outputs/embeddings/ (MERT-v1-95M/ vs MERT-v1-330M/)
- [x] Model naming scheme: spotilyzer_model_{embedder}_{date}.joblib
- [x] 8738-sample dataset (DE, US, UK, FR, BR, ES charts + genre clusters)
- [x] 330M model trained and evaluated (MERTv1330M_20260317)
- [x] MODEL_COMPARISON.md cheat sheet created
- [x] Chart discovery via analyze_clusters.py performed
- [x] recon_clusters.py + clusters_recon.yaml created
- [x] Chart categorization: 22 validated, 2 suspicious (KR/TH), 7 excluded (AR/CL/PT/TR/AE/NZ/IN)
- [x] scout_kworb.py: Kworb _weekly_totals, 6 markets, checkpoint system, miss cache, ISRC cache
- [x] Model deployed: spotilyzer_model_MERTv1330M_main+spotify_charts+kworb_validated_20260319.joblib

### Long-term
- [ ] YouTube views as third source
- [ ] Genre-specific models
- [ ] Test on AI-generated tracks (Mureka, Suno)

---

## Hardware

**Current:** GTX 1660 Ti (6 GB VRAM)
**Planned:** Upgrade to 16+ GB

**Relevance for training:**
- MERT embedding computation: ~2 GB VRAM
- XGBoost/LightGBM training: CPU-based, VRAM irrelevant
- UMAP visualization: CPU, RAM-intensive for large datasets

---

## References

### Obsidian Reference System (from Session 3)

Path: `D:\Software\Tools\Obsidian Vaults\AV-Obsidian\Projekte\Spotilyzer\`

| File/Folder | Content |
|---|---|
| `Master.md` | Central navigation (indices, curated docs, logs) |
| `Reference_Docs\curated\2026-03-18\Chart-Datenquellen_für_Modul-System.md` | ⭐ Working basis for kworb_deezer |
| `Indices\2026-03-18\` | Outlines of three original research pieces (ChatGPT × 2, Gemini) |
| `Reference_Docs\original\2026-03-18\` | Complete ChatGPT/Gemini deep-dive outputs |

### Project Documents (Main Project)
- `Spotilyzer/CLAUDE.md` — Main project documentation
- `Spotilyzer/!BU/Spotilyzer_GenAI_Encoder_Analysis.md` — CLAP/HeartCLAP analysis
- `Spotilyzer/!BU/UVR_Index_for_Spotilyzer.md` — Stem separation options

### External
- [Last.fm API Docs](https://www.last.fm/api)
- [pylast (Python Last.fm Client)](https://github.com/pylast/pylast)
- [Deezer API Docs](https://developers.deezer.com/api)
- [Kworb.net](https://kworb.net) — Spotify Top 200 Charts, daily/cumulative
- [MusicBrainz API](https://musicbrainz.org/doc/MusicBrainz_API) — ISRC lookup (1 req/s)
- [XGBoost sample_weight](https://xgboost.readthedocs.io/en/stable/python/python_api.html)
- [mutagen (ID3 tagging)](https://mutagen.readthedocs.io/)
