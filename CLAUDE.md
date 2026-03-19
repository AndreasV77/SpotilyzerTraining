# CLAUDE.md — SpotilyzerTraining

Arbeitsdokument für das Modell-Training-Subprojekt von Spotilyzer.

**Erstellt:** 2026-03-07
**Zuletzt aktualisiert:** 2026-03-19 (Session 5: kworb-Modul implementiert, scout_kworb.py + Checkpoint-System, Datensatz auf ~8960 validated / ~3700 Hits erweitert, neues Modell BA=63.0% / Hit R.=72.8%)

**Wichtige Regel:** CLAUDE.md immer nach abgeschlossenen Schritten aktualisieren — nie auf Basis von laufenden oder geplanten Ergebnissen schreiben. Metriken immer aus Reports lesen, nicht schätzen.

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
Copy-Item outputs/models/spotilyzer_model_MERTv1330M_*.joblib ..\Spotilyzer\models\
Copy-Item outputs/reports/training_report_MERTv1330M_*.json ..\Spotilyzer\models\
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

### Aktueller Modellstand (Stand 2026-03-19, Quelle: evaluation_reports)

Alle Metriken auf echtem Holdout-Set (20%). Datensatz: validated-only.

| Modell | Datensatz | Holdout | BA | Hit R. | Flop R. | Status |
|--------|-----------|---------|-----|--------|---------|--------|
| `MERTv1330M_main+spotify_charts+kworb_validated_20260319` | ~8960 val. | 1173 | **63.0%** | **72.8%** | 68.7% | **Aktiv** |
| `MERTv1330M_main+spotify_charts_validated_20260319` | 5660 val. | 1132 | 60.9% | 55.1% | 69.2% | Vorgänger |
| `MERTv195M_main+spotify_charts_validated_20260319` | 5660 val. | 1132 | 57.4% | 47.7% | 68.7% | Vorgänger |
| `MERTv1330M_validated_20260318` | 5262 val. | 967 | 57.5% | 37.5% | 71.1% | Vorgänger |
| `MERTv195M_validated_20260318` | 5262 val. | 967 | 53.2% | 27.3% | 68.9% | Vorgänger |
| `MERTv195M_origparams_validated_20260318` | 5262 val. | 967 | 52.6% | 24.8% | 69.2% | Referenz |

**Session-5-Befund:** kworb-Modul (Kworb.net _weekly_totals, 6 Märkte) lieferte 2738 neue Tracks, 2497 Hits → Hit-Count von 1216 auf ~3700 verdreifacht. Hit Recall 330M: 55.1% → 72.8% (+17.7pp). Trend stabil: je +600 Hits → je +17–18pp Hit Recall. Kworb-Track-IDs waren bereits alle in Embeddings vorhanden (populäre Tracks vom Deezer-Scouting bereits erfasst). Confusion: 137 Hits als Mid klassifiziert — Mid-Klasse bleibt die größte Fehlerquelle.

**Session-4-Befund:** spotify_charts-Modul lieferte 960 neue Tracks, 579 Hits → Hit-Count von 637 auf 1216 fast verdoppelt. Hit Recall 330M: 37.5% → 55.1% (+17.6pp). Hypothese bestätigt: reines Datenproblem, kein Hyperparameter-Problem.

**Ursachenanalyse der früheren 26% Flop Recall:** 3900 "contested" Tracks (Deezer/Last.fm Widerspruch) wurden alle als "mid" gelabelt → Mid von 2114 auf 6032 aufgebläht (3×). Behobener Datensatz via `--validated-only`.

**Parameter-Befund (95M):** Tuned vs. origparams → marginaler Unterschied (+0.6% BA). Bei größerem Datensatz könnte origparams kompetitiver sein — Testregel vorerst ausgesetzt, da Datenmenge Priorität hat.

**Strategische Konsequenz:** Hit Recall 72.8% — noch 7.2pp bis Ziel ≥80%. Nächster Schritt: weiteres Datenwachstum (mehr Märkte in Kworb, neue Spotify-Charts-Snapshots) oder Hyperparameter-Tuning.

### Aktueller Datensatz-Stand (2026-03-19, Session 5)

Kombinierter Datensatz: Haupt-JSONL (Deezer-Scouting) + spotify_charts-Modul + kworb-Modul

| Quelle | Tracks | Validated | Hits (val.) | Embeddings |
|--------|--------|-----------|-------------|------------|
| main (Deezer) | 9.661 | 5.262 | 637 | 8.794 |
| spotify_charts | 960 | 960 | 579 | 960 |
| kworb | 2.738 | 2.738 | 2.497 | 2.738 (bereits in main) |
| **Gesamt (dedup)** | **~13.100** | **~8.960** | **~3.713** | **9.526** |

**Holdout-Set (Session 5):** 1173 Samples (415 Flops, 298 Hits, 460 Mids) — 20% aus ~8960 validated

**Spotify Charts abgedeckt (2026-03-19):**
- `regional-{us/gb/de/jp/br/mx/global}-weekly-2026-03-12.csv`
- Pfad: `G:/Dev/SpotilyzerData/spotify/2026-03-19/`
- Match-Rate: 978/994 (98.4%) via Deezer-Suche; 16 Misses (vermutlich JP-Kanji)

**Kworb abgedeckt (2026-03-19):**
- Märkte: us, gb, de, jp, br, mx — `_weekly_totals` (kumulierte Historie seit 2013)
- Filter: Total ≥ 20.000.000 Streams → 3000 Top-Tracks nach Dedup
- Match-Rate: 2738/3000 (91.3%) via Deezer-Suche; 262 Misses (Spotify-Exklusives / regionale Lücken)
- ISRC: `--skip-mb` (alle via Artist+Title-Suche); `enrich_isrc.py` für spätere ISRC-Anreicherung geplant

**Nächster Schritt:** Weitere Datenwachstums-Optionen (neue Spotify-Charts-Snapshots, ggf. weitere Kworb-Märkte) oder Hyperparameter-Tuning für die letzten 7.2pp bis Hit Recall ≥80%.

---

## Datenstruktur (SpotilyzerData)

**Speicherort:** `G:/Dev/SpotilyzerData` (außerhalb des Repos, zu groß für Git)

```
G:/Dev/SpotilyzerData/
├── previews/                      # Audio-Dateien (geteilt über alle Datensätze)
│   ├── 00/ ... ff/                # MD5-Hash-Sharding (256 Ordner)
│   │   └── {track_id}.mp3         # Deezer-ID als Dateiname
│   └── ...
│
├── metadata/
│   └── tracks.jsonl               # Haupt-Datensatz (Deezer-Scouting)
│
├── datasets/                      # Modul-Datensätze (separate JSONL je Modul)
│   ├── spotify_charts/
│   │   └── tracks.jsonl           # Spotify Top 200 Charts
│   └── kworb/
│       ├── tracks.jsonl           # Kworb _weekly_totals (kumulierte Historie)
│       ├── isrc_cache.json        # MusicBrainz ISRC-Cache
│       └── deezer_miss_cache.json # Tracks ohne Deezer-Match (Resume-Skip)
│
├── spotify/                       # Rohe Spotify Chart CSVs (manuell gezogen)
│   └── {YYYY-MM-DD}/
│       └── regional-{country}-weekly-{date}.csv
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
│   ├── clusters_recon.yaml      # Chart-Kategorisierung für Recon (siehe Chart-Expansion-Sektion)
│   ├── paths.yaml               # Pfade (Preview-Speicherort etc.)
│   ├── thresholds.yaml          # Rank/Plays-Schwellenwerte für Labels
│   └── training.yaml            # Hyperparameter für XGBoost
│
├── scripts/
│   ├── run_pipeline.py          # Orchestrierungs-Skript (Haupteinstieg)
│   ├── scout_deezer.py          # Deezer-Scouting (Genre-Cluster + Charts)
│   ├── scout_spotify.py         # Spotify Charts CSV → datasets/spotify_charts/tracks.jsonl
│   ├── download_previews.py     # Preview-Download (mit ID3-Tagging + Sharding)
│   ├── enrich_lastfm.py         # Last.fm-Anreicherung
│   ├── compute_labels.py        # Multi-Source-Label-Berechnung
│   ├── extract_embeddings.py    # MERT-Embedding-Extraktion
│   ├── train_model.py           # XGBoost-Training mit Sample Weights
│   ├── evaluate.py              # Metriken + Confusion Matrix (Holdout-Set aus Bundle)
│   ├── inspect_dataset.py       # Read-only Diagnose-Tool (Label-Verteilung, Robustheit, etc.)
│   ├── analyze_clusters.py      # Cluster-Analyse: Sanity-Check, Stats, Overlap, Chart-Discovery
│   ├── recon_clusters.py        # Cluster-Recon: Vorprüfung bekannter Cluster (Aktualität, Spam, Overlap) — VOR Scouting
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
    ├── recon/                   # Recon-Track-Listen (ohne Preview-URLs)
    │   └── tracks_recon_TIMESTAMP.jsonl
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

# Einzelne Skripte direkt (mit Modul-Datensatz):
python scripts/extract_embeddings.py --model 95M --dataset spotify_charts --append
python scripts/train_model.py --embedder 95M --dataset main spotify_charts --validated-only
python scripts/evaluate.py --embedder 95M --dataset main spotify_charts --validated-only --save-report

# Ohne Modul-Datensatz (nur main):
python scripts/train_model.py --embedder 95M --validated-only
python scripts/evaluate.py --embedder 95M --validated-only --save-report

# Explizites Modell evaluieren (wenn Autodetect nicht greift):
python scripts/evaluate.py --model outputs/models/spotilyzer_model_MERTv195M_origparams_validated_20260318.joblib --embedder 95M --validated-only --save-report

# Spotify Charts scouten (neuer Snapshot):
python scripts/scout_spotify.py --input-dir G:/Dev/SpotilyzerData/spotify/YYYY-MM-DD --dry-run
python scripts/scout_spotify.py --input-dir G:/Dev/SpotilyzerData/spotify/YYYY-MM-DD

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

**Typischer Einsatz:** Liest aus `clusters.yaml` (Training-Config) + `tracks.jsonl`. Für `--cluster-stats`/`--overlap` muss also ein Scouting-Lauf bereits erfolgt sein.

**Hinweis zu `--label-distribution`:** Führt intern denselben Codepfad aus wie `--cluster-stats` — kein Unterschied im Output. Status unklar (möglicherweise als eigenständiger Pfad geplant, aber nicht implementiert).

---

### recon_clusters.py — Cluster-Vorprüfung

Reconnaissance-Tool für bekannte Chart-Cluster. Liest aus `configs/clusters_recon.yaml` (NICHT `clusters.yaml`).

**Was es tut:**
- Track-Count, Rank-Verteilung (Min/Max/Median/P25/P75), Artist-Diversity
- Release-Dates via Album-API — nur für 15 Sample-Tracks (Top 5 / Mid 5 / Bottom 5)
- Overlap-Analyse zwischen Charts
- Spam-Detection (Einzelkünstler-Dominanz, alte Releases, Nischen-Ranks)

**Was es NICHT tut:** Preview-URLs, Downloads, `tracks.jsonl`-Schreibzugriff, Last.fm

```powershell
# Default: validated + suspicious Charts
python scripts/recon_clusters.py

# Nur validierte Charts
python scripts/recon_clusters.py --scope validated

# Alle inkl. excluded (vollständige Dokumentation)
python scripts/recon_clusters.py --scope all

# Bestimmte Charts (aus beliebiger Kategorie)
python scripts/recon_clusters.py --charts KR AR CL

# Ad-hoc: Chart ohne Config-Eintrag testen
python scripts/recon_clusters.py --add-chart VN 1234567890 "Vietnam"

# Dry-Run: zeigt Charts + geschätzte API-Calls
python scripts/recon_clusters.py --dry-run
```

**Output:**
- `outputs/reports/recon_TIMESTAMP.json` — Statistiken + Warnings
- `outputs/recon/tracks_recon_TIMESTAMP.jsonl` — Track-Liste (ohne Preview-URLs)

**`clusters_recon.yaml` — Kategorien:**

| Kategorie | Beschreibung |
|-----------|--------------|
| `existing` | Bereits in `clusters.yaml` für Training konfiguriert |
| `validated` | Offiziell (Deezer Charts), aktuell, keine Auffälligkeiten |
| `suspicious` | Potenziell manipuliert — erfordert manuelle Entscheidung |
| `excluded` | Nicht brauchbar (veraltet, user-kuratiert, API-Bug) |

**Bekannte Lücke:** Die `existing`-Charts (DE, US, UK, FR, BR, ES, JP, GLOBAL) haben in `clusters_recon.yaml` keine `playlist_id` — sie werden vom Recon übersprungen, auch bei `--scope all`. Sie sollten für eine vollständige Analyse mit Playlist-IDs ergänzt werden.

**Spam-Detection-Schwellenwerte** (aus `recon_settings` in `clusters_recon.yaml`):
- Einzelkünstler-Dominanz > 30% → Warnung
- Artist-Diversity < 0.5 → Warnung
- < 30% Releases aus letzten 12 Monaten → "Chart veraltet?"
- Rank-Median > 900.000 → "Nischen-Content?"

---

### Workflow: Cluster-Erweiterungsplanung

Dieser Workflow ist **Voraussetzung** für jeden neuen Scouting-Lauf mit erweiterten Clustern. Er läuft vor `scout_deezer.py` und ist von der normalen Trainings-Pipeline getrennt.

```
1. analyze_clusters.py --discover-charts
      → findet Playlist-IDs für neue Länder via Deezer-Suche
      → gibt YAML-Snippet für clusters_recon.yaml aus
      ↓
2. clusters_recon.yaml aktualisieren
      → neue Einträge in validated/suspicious eintragen
      → (manuell / durch Claude im Chat)
      ↓
3. recon_clusters.py
      → Vorprüfung: Aktualität, Spam, Artist-Diversity, Overlap
      → Report in outputs/reports/recon_TIMESTAMP.json
      ↓
4. Entscheidung: suspicious → validated oder excluded
      → clusters_recon.yaml manuell aktualisieren
      ↓
5. clusters.yaml aktualisieren
      → validated Cluster eintragen (falls für Scouting vorgesehen)
      ↓
6. analyze_clusters.py --sanity
      → prüft ob alle playlist_ids in clusters.yaml erreichbar sind
      ↓
7. Cluster-Planung
      → welche Cluster fürs Scouting? Tier-Einteilung? Gewichtung?
      → Basis: recon-Report + eigene Einschätzung
      → Tier-System: siehe Chart-Expansion-Sektion unten
      ↓
8. scout_deezer.py
      → Scouting ausschließlich für beschlossene Cluster
      ↓
9. analyze_clusters.py --cluster-stats --overlap
      → Post-Scouting-Analyse (braucht tracks.jsonl)
```

---

## Chart-Expansion: Stand und Entscheidungen

**Stand:** 2026-03-18
**Referenzdokumente:** 
- `outputs/reports/recon_*.json` — Recon-Reports
- `outputs/recon/tracks_recon_*.jsonl` — Sample-Track-Listen
- Claude.ai Project: Konsolidierter Index aus Chat-Logs (Obsidian workbench: `1_Continue_*.md`, `2_Datenquellen_*.md`, `3_Chat-Verlauf_*.md`)

### Hintergrund: Warum Chart-Expansion?

**Kernproblem:** Zu wenige Hit-Samples (623 von 4813 = 12.9%). FR/BR/ES-Charts lieferten fast nur Mids/Flops (+10 Hits, +645 Flops, +871 Mids). Hit Recall stagniert bei 27–30%.

**Ziel:** ≥2000 Hit-Samples durch systematische Chart-Erweiterung.

### Discovery: Deezer Chart-Infrastruktur

**Erkenntnis 1: "Deezer Charts" Account**
- Account-ID: **637006841** (nicht die Editorial-ID 2!)
- Semi-offizieller Account mit automatisiert generierten Länder-Charts
- Alle Charts haben ~100 Tracks, werden regelmäßig aktualisiert

**Erkenntnis 2: Search-API-Bug**
- Die Deezer Search-API liefert **0 Follower** für diese Playlists
- Echte Follower-Zahlen nur via direktem Playlist-API-Call (`/playlist/{id}`)
- Beispiel: "Top Italy" zeigt 0 Follower in Search, aber 678.285 via Playlist-API

**Erkenntnis 3: Release-Date-Limitation**
- Playlist-Track-API liefert kein `release_date` im Album-Objekt
- Workaround in `recon_clusters.py`: Album-API-Call für 15 Sample-Tracks (Top 5 / Mid 5 / Bottom 5)

### Chart-Kategorisierung (vollständig)

#### Existierend (8) — bereits in `clusters.yaml`

| Code | Name | Status |
|------|------|--------|
| DE | Germany | Aktiv |
| US | United States | Aktiv |
| UK | United Kingdom | Aktiv |
| FR | France | Aktiv |
| JP | Japan | Aktiv |
| BR | Brazil | Aktiv |
| ES | Spain | Aktiv |
| GLOBAL | Worldwide | Aktiv |

#### Validiert (22) — bereit für Integration

**Europa:**
| Code | Playlist-ID | Follower | Bemerkung |
|------|-------------|----------|-----------|
| IT | 1116187241 | 678K | Aktuell, Bruno Mars / Alex Warren |
| NL | 1266971851 | 273K | Aktuell |
| SE | 1313620305 | 69K | Aktuell |
| AT | 1313615765 | 61K | Aktuell |
| CH | 1313617925 | 58K | Aktuell |
| BE | 1266968331 | 152K | Aktuell, Taylor Swift |
| NO | 1313619885 | 15K | Aktuell |
| DK | 1313618905 | 32K | Aktuell |
| FI | 1221034071 | 56K | Lokale Künstler! |
| IE | 1313619455 | 39K | Aktuell |
| PL | 1266972311 | 107K | Aktuell |

**Amerika:**
| Code | Playlist-ID | Follower | Bemerkung |
|------|-------------|----------|-----------|
| MX | 1111142361 | 1.05M | Latin Hot, Peso Pluma / Bad Bunny |
| CA | 1652248171 | 42K | Aktuell |
| CO | 1116188451 | 1.5M | Latin aktuell, Ryan Castro |

**Asien-Pazifik:**
| Code | Playlist-ID | Follower | Bemerkung |
|------|-------------|----------|-----------|
| AU | 1313616925 | 59K | Aktuell |
| ID | 1116188761 | 338K | Aktuell |
| PH | 1362518895 | 57K | Aktuell |
| SG | 1313620765 | 21K | Aktuell |
| MY | 1362515675 | 5K | Lokale Acts |

**MENA:**
| Code | Playlist-ID | Follower | Bemerkung |
|------|-------------|----------|-----------|
| EG | 1362501615 | 111K | MENA-Markt |
| SA | 1362521285 | 27K | MENA-Markt |
| ZA | 1362528775 | 62K | Aktuell |

#### Fragwürdig (5) — manuelle Prüfung erforderlich

| Code | Problem | Sample-Tracks |
|------|---------|---------------|
| KR | Klassik-Orchester auf #2/#3 — Bot-Manipulation? | Borodine, Saint-Saëns statt K-Pop |
| AR | Nur BTS/Jimin — K-Pop Stan Takeover | "Who", "Set Me Free", "Let Me Know" |
| CL | NUR alte BTS-Tracks (2014!) — definitiv manipuliert | "Danger", "24/7=Heaven" |
| TH | Seltsamer Mix — französische Star Academy auf #3? | Ungewöhnliche Genre-Mischung |
| PT | "Barulho Para Relaxar" = White-Noise-Tracks | Kim Wilde "You Came" (1988) |

**Entscheidung ausstehend:** Diese Charts könnten trotzdem brauchbare Tracks enthalten, wenn man die Top-Positionen ignoriert. Erfordert manuellen Recon-Lauf mit `--charts KR AR CL TH PT` und Einzelprüfung.

#### Ausgeschlossen (4) — nicht brauchbar

| Code | Grund | Detail |
|------|-------|--------|
| TR | Veraltet | "Top Turkey **2020**" — 6 Jahre alt |
| AE/UAE | User-kuratiert | 2019, nur 7 Follower |
| NZ | User-kuratiert | 293 Tracks, keine echte Chart |
| IN | API-Bug | Suche liefert Indonesia statt India |

#### Nicht durchsucht (10) — Status unklar

CN, RU, VN, TW, HK, IL, GR, CZ, HU, RO — in früheren Sessions als problematisch markiert, Begründung nicht mehr nachvollziehbar. Bei Bedarf erneut prüfen.

### Geplantes Tier-System

**Konzept:** Charts nach Marktrelevanz gewichten. Noch nicht implementiert — finale Einteilung erfolgt nach Recon-Daten-Analyse (Overlap, Rank-Verteilung).

| Tier | Gewicht | Kriterien | Kandidaten |
|------|---------|-----------|------------|
| **Tier 1** | 1.0 | Internationale Referenz, definiert Mainstream | US, UK, GLOBAL |
| **Tier 2** | 0.85 | Große Export-Märkte, signifikanter Einfluss | DE, FR, AU, CA, JP, BR |
| **Tier 3** | 0.7 | Mittelgroße Märkte, eigene Szene | ES, IT, MX, NL, SE, KR (falls validiert) |
| **Tier 4** | 0.5 | Lokale Märkte, Nischen-Relevanz | PL, AT, CH, BE, NO, DK, FI, IE, etc. |

**Anwendung (geplant):**
- Track in mehreren Charts → Durchschnitt der Tier-Gewichte
- Track nur in Tier-4-Chart → `robustness * 0.5`
- Implementierung in `thresholds.yaml` oder `clusters.yaml` (noch zu entscheiden)

**Wichtig:** "Gewichtungen sind Vermutungen mit Krawatte." Das Tier-System ist eine Heuristik, keine wissenschaftlich validierte Metrik. Transparenz über Unsicherheit hat Vorrang vor Pseudo-Präzision.

### Spam-Detection-Kriterien (in recon_clusters.py)

| Kriterium | Schwellenwert | Bedeutung |
|-----------|---------------|-----------|
| Einzelkünstler-Dominanz | > 30% | Ein Artist dominiert den Chart → Streaming-Farm? |
| Artist-Diversity | < 0.5 | Wenige unique Artists / Total Tracks |
| Release-Aktualität | < 30% aus 12 Mo. | Chart veraltet? |
| Rank-Median | > 900.000 | Nischen-Content statt Mainstream-Chart |

### Status Deezer Chart-Erweiterung (2026-03-19)

**Entscheidung:** Deezer-Chart-Erweiterung wird **nicht weiter priorisiert**. Erfahrung zeigt, dass zusätzliche Deezer-Länder-Charts überwiegend Mids/Flops liefern, kaum Hits. Stattdessen: `kworb_deezer`-Modul.

Verbleibende offene Punkte (nur bei Bedarf):
1. `clusters_recon.yaml`: Playlist-IDs für DE/US/UK/FR/BR/ES/JP/GLOBAL ergänzen (→ Recon überspringt sie aktuell)
2. KR, TH: Gezielter Recon-Lauf — nach kworb_deezer-Implementierung entscheiden ob relevant
3. 22 validated Charts ggf. als `robustness`-Signal in kworb_deezer nutzen (nicht als Primärquelle)

### Externe Chart-Quellen (Primärstrategie ab Session 3)

Deezer-Charts haben eine harte Decke. Primärstrategie ist jetzt das `kworb_deezer`-Modul.

**Hauptquellen (für kworb_deezer Phase 1):**

| Quelle | Zugang | Format | Märkte |
|--------|--------|--------|--------|
| **Kworb.net** | Scraping, kein Login | Statisches HTML, `pandas.read_html()` | ~70 Länder, Spotify Top 200 |
| **charts.spotify.com** | Manueller Download, Login erforderlich | CSV pro Land/Woche | ~70 Länder |
| **MusicBrainz** | API (1 req/s), kostenlos | JSON | ISRC-Lookup für Deduplizierung |

**Ergänzende Quellen (Phase 2/3):**

| Quelle | Zugang | Mehrwert |
|--------|--------|----------|
| Billboard Japan | CSV-Download, kein Login | J-Pop ohne Scraping |
| Hung Medien Network | Scraping (konsistentes Schema) | 15 EU-Länder + Ozeanien |
| Zertifizierungs-DBs (BVMI, BPI, RIAA, etc.) | Öffentlich durchsuchbar | `robustness=validated` Signal |

**Nicht verwendbar:**

| Quelle | Grund |
|--------|-------|
| Spotify API | Keine Stream-Counts; ToS verbietet Scraping |
| Apple Music | Keine Playcount-Daten öffentlich |
| Shazam | Keine öffentliche API seit 2019 |

---

### Abhängigkeiten zwischen Schritten

```
Haupt-Pipeline (Deezer):
1. scout_deezer.py
    ↓ metadata/tracks.jsonl (initial: track_id, title, artist, album, clusters, deezer_rank)
2. download_previews.py [--dataset main]
    ↓ previews/{shard}/{track_id}.mp3 (mit ID3-Tags, MD5-Sharding)
    ↓ metadata/tracks.jsonl (file_path hinzugefügt)
3. enrich_lastfm.py
    ↓ metadata/tracks.jsonl (lastfm_* Felder hinzugefügt)
4. compute_labels.py
    ↓ metadata/tracks.jsonl (label + robustness hinzugefügt)

Modul-Pipeline (spotify_charts):
1b. scout_spotify.py --input-dir G:/Dev/SpotilyzerData/spotify/YYYY-MM-DD
    ↓ datasets/spotify_charts/tracks.jsonl (track_id, label=hit/mid, robustness=validated)
2b. download_previews.py --dataset spotify_charts
    ↓ previews/{shard}/{track_id}.mp3 (geteilt mit Haupt-Pipeline!)
    ↓ datasets/spotify_charts/tracks.jsonl (file_path hinzugefügt)

Modul-Pipeline (kworb):
1c. scout_kworb.py --min-streams 20000000 --max-tracks 3000 --skip-mb
    ↓ datasets/kworb/tracks.jsonl (chart_entries, chart_score, label, robustness=validated)
    ↓ datasets/kworb/isrc_cache.json + deezer_miss_cache.json (Checkpoint-System)
    (--skip-mb: MusicBrainz überspringen, direkt Deezer-Suche)
    (Checkpoint alle 100 Tracks: kworb_checkpoint.jsonl, bei Abschluss gelöscht)
2c. download_previews.py --dataset kworb
    ↓ previews/{shard}/{track_id}.mp3 (geteilt, meist bereits vorhanden!)
    ↓ datasets/kworb/tracks.jsonl (file_path hinzugefügt)

Gemeinsame Pipeline (ab Embeddings):
5. extract_embeddings.py [--model 95M|330M] [--dataset spotify_charts --append]
    ↓ outputs/embeddings/MERT-v1-{version}/embeddings.npy + embeddings_meta.csv + embeddings_info.json
    (Checkpoint/Resume: --resume Flag, speichert alle 500 Tracks)
    (--append: neue Tracks zu bestehendem .npy hinzufügen)
6. train_model.py [--embedder 95M|330M] [--dataset main spotify_charts] [--validated-only]
    ↓ outputs/models/spotilyzer_model_{tag}[_{exp_label}][_{datasets}][_validated]_{date}.joblib
    ↓ outputs/reports/training_report_{tag}_{datasets}_{date}.json
    (Sample weights: compute_sample_weight("balanced") × robustness weights)
    (test_track_ids werden im Bundle gespeichert → Holdout-Evaluation in evaluate.py)
    (Per-Embedder-Params aus training.yaml: models.MERT-v1-95M / models.MERT-v1-330M)
7. evaluate.py [--embedder 95M|330M] [--dataset main spotify_charts] [--validated-only] [--save-report]
    ↓ outputs/reports/evaluation_report_{model_suffix}.json
    (Testet nur auf Holdout-Set aus Bundle — nicht auf Trainingsdaten!)
    (Autodetect wählt neuestes *validated*.joblib für den Embedder)
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

Alle Werte auf echtem Holdout-Set (20%). Quelle: `evaluation_report_*.json`

### Session 5 — main + spotify_charts + kworb (~8960 validated, 1173 Holdout)

| Metrik | 330M | Ziel |
|--------|------|------|
| Flop Recall | **68.7%** ✓ | ≥ 50% |
| Hit Recall | **72.8%** ✗ | ≥ 80% |
| Balanced Accuracy | **63.0%** ✗ | ≥ 65% |

### Session 4 — main + spotify_charts (5660 validated, 1132 Holdout) — Referenz

| Metrik | 95M | 330M | Ziel |
|--------|-----|------|------|
| Flop Recall | 68.7% ✓ | 69.2% ✓ | ≥ 50% |
| Hit Recall | 47.7% ✗ | 55.1% ✗ | ≥ 80% |
| Balanced Accuracy | 57.4% ✗ | 60.9% ✗ | ≥ 65% |

### Session 3 — main only (5262 validated, 967 Holdout) — Referenz

| Metrik | 95M_orig | 95M_tuned | 330M_tuned |
|--------|----------|-----------|------------|
| Flop Recall | 69.2% | 68.9% | 71.1% |
| Hit Recall | 24.8% | 27.3% | 37.5% |
| Balanced Accuracy | 52.6% | 53.2% | 57.5% |

**Flop Recall-Ziel erreicht.** Hit Recall: je +~2500 Hits → +17–18pp. Trend stabil über 3 Sessions. Letzter Schritt bis ≥80%: weiteres Datenwachstum oder Hyperparameter-Tuning.

---

## Offene Aufgaben

### Kurzfristig (nächste Session)
- [x] ~~95M Embeddings extrahieren~~ ✅ (2026-03-17, 8738 Samples)
- [x] ~~95M Neutraining~~ ✅ (MERTv195M_20260317, BA=47.8% — schlechter als 330M)
- [x] ~~Recon-Lauf~~ ✅ (2026-03-18, alle validated + suspicious Charts geprüft)
- [x] ~~Suspicious-Entscheidungen~~ ✅ AR/CL/PT → excluded; KR/TH bleiben suspicious
- [x] ~~Scouting-Lauf~~ ✅ (2026-03-18, --min-rank 600000, bestehende Cluster)
- [x] ~~Embeddings --append~~ ✅ (2026-03-18, 56 neue Tracks, beide Modelle)
- [x] ~~Training auf neuem Datensatz~~ ✅ (2026-03-18, alle drei Modelle, --validated-only)
- [x] ~~spotify_charts-Modul~~ ✅ (2026-03-19, scout_spotify.py + --dataset-Flag in allen Skripten)
- [x] ~~Training + Eval auf main+spotify_charts~~ ✅ (2026-03-19, 330M: BA=60.9%, Hit R.=55.1%)
- [x] ~~evaluate.py --dataset-Flag + Autodetect-Fix~~ ✅ (2026-03-19)
- [ ] `models/MODEL_COMPARISON.md` in Spotilyzer aktualisieren (Session-5-Metriken eintragen)
- [ ] `compute_labels.py` Bug 3 fixen: Dissent-Logik schickt Widersprüche zu "mid" statt "contested"
- [x] ~~kworb-Modul implementieren~~ ✅ (scout_kworb.py + Checkpoint-System, 2026-03-19)
- [x] ~~Training + Eval auf main+spotify_charts+kworb~~ ✅ (330M: BA=63.0%, Hit R.=72.8%, 2026-03-19)

### Modul-System: Kworb-Scraper (abgeschlossen)

**Status:** ✅ scout_kworb.py implementiert und erfolgreich gelaufen (2026-03-19).

**Ergebnis:** 2738 Tracks, 2497 Hits, alle Embeddings bereits vorhanden. Training lieferte BA=63.0%, Hit R.=72.8%.

**Offene Todos (nice-to-have):**
- [ ] `enrich_isrc.py` — Background-Skript: ISRC für `isrc: null`-Tracks via MusicBrainz nachfüllen (aktuell `--skip-mb` genutzt)
- [ ] `configs/datasets/kworb.yaml` — Markt-Liste, Tier-Gewichte, Hit-Thresholds (aktuell hardcoded in scout_kworb.py)

### Cluster-Erweiterungsplanung (Deezer — niedrige Priorität)
- [x] ~~`recon_clusters.py` laufen lassen~~ ✅ (2026-03-18)
- [x] ~~AR, CL, PT~~ ✅ → excluded (manipuliert/Spam)
- [ ] KR, TH: gezielter Recon-Lauf (`--charts KR TH`) → dann entscheiden (nach kworb_deezer)
- [ ] `clusters_recon.yaml`: DE/US/UK/FR/BR/ES/JP/GLOBAL mit `playlist_id` ergänzen
- [ ] Tier-Einteilung auf Basis Overlap/Rank-Daten finalisieren

### Mittelfristig
- [x] ~~Mehr Hit-Samples: Ziel ≥2000 validated Hits~~ ✅ (~3700 Hits, Session 5)
- [ ] Genre-balanced Sampling evaluieren
- [ ] LightGBM als Alternative testen
- [ ] `configs/thresholds.yaml` — Last.fm-Schwellenwerte kalibrieren (oder via Modul-System obsolet)
- [ ] Bestehende CSV-Daten in `scout_results/` → JSONL migrieren (einmalig, optional)

### Erledigt
- [x] spotify_charts-Modul: scout_spotify.py, download_previews.py --dataset, extract_embeddings.py --dataset, train_model.py --dataset, evaluate.py --dataset
- [x] evaluate.py Autodetect-Fix: Glob *validated* statt _validated_ (matcht main+spotify_charts)
- [x] JSONL-Refactoring (statt CSV/pandas)
- [x] MD5-Sharding für Previews
- [x] 7 neue Genre-Cluster (23 gesamt)
- [x] Radio-Scouting in `scout_deezer.py`
- [x] `scripts/utils/` mit `paths.py`, `playlist.py`, `metadata.py`
- [x] Label-swap-bug fix (alphabetical LabelEncoder → target_names korrekt)
- [x] compute_sample_weight("balanced") × robustness weights
- [x] Embedding-Checkpoint/Resume-System (--resume, alle 500 Tracks)
- [x] `--append`-Flag in `extract_embeddings.py` (nur neue Tracks embedden, bestehende überspringen)
- [x] Modell-Auswahl in run_pipeline.py (interaktives Menü + --model CLI-Flag)
- [x] `--embedder`-Flag in train_model.py und evaluate.py
- [x] Embedder-Unterordner in outputs/embeddings/ (MERT-v1-95M/ vs MERT-v1-330M/)
- [x] Modell-Naming-Schema: spotilyzer_model_{embedder}_{date}.joblib
- [x] 8738-Sample-Datensatz (DE, US, UK, FR, BR, ES Charts + Genre-Cluster)
- [x] 330M-Modell trainiert und evaluiert (MERTv1330M_20260317)
- [x] MODEL_COMPARISON.md Cheat-Sheet erstellt
- [x] Chart-Discovery via analyze_clusters.py durchgeführt
- [x] recon_clusters.py + clusters_recon.yaml erstellt
- [x] Chart-Kategorisierung: 22 validated, 2 suspicious (KR/TH), 7 excluded (AR/CL/PT/TR/AE/NZ/IN)
- [x] scout_kworb.py: Kworb _weekly_totals, 6 Märkte, Checkpoint-System, Miss-Cache, ISRC-Cache
- [x] Modell deployed: spotilyzer_model_MERTv1330M_main+spotify_charts+kworb_validated_20260319.joblib

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

### Obsidian-Referenzsystem (ab Session 3)

Pfad: `D:\Software\Tools\Obsidian Vaults\AV-Obsidian\Projekte\Spotilyzer\`

| Datei/Ordner | Inhalt |
|---|---|
| `Master.md` | Zentrale Navigation (Indices, curated Docs, Logs) |
| `Reference_Docs\curated\2026-03-18\Chart-Datenquellen_für_Modul-System.md` | ⭐ Arbeitsgrundlage kworb_deezer |
| `Indices\2026-03-18\` | Gliederungen der drei Original-Recherchen (ChatGPT × 2, Gemini) |
| `Reference_Docs\original\2026-03-18\` | Vollständige ChatGPT/Gemini Deep-Dive-Outputs |

### Projekt-Dokumente (Hauptprojekt)
- `Spotilyzer/CLAUDE.md` — Hauptprojekt-Dokumentation
- `Spotilyzer/!BU/Spotilyzer_GenAI_Encoder_Analysis.md` — CLAP/HeartCLAP-Analyse
- `Spotilyzer/!BU/UVR_Index_for_Spotilyzer.md` — Stem-Separation-Optionen

### Externe
- [Last.fm API Docs](https://www.last.fm/api)
- [pylast (Python Last.fm Client)](https://github.com/pylast/pylast)
- [Deezer API Docs](https://developers.deezer.com/api)
- [Kworb.net](https://kworb.net) — Spotify Top 200 Charts, täglich/kumuliert
- [MusicBrainz API](https://musicbrainz.org/doc/MusicBrainz_API) — ISRC-Lookup (1 req/s)
- [XGBoost sample_weight](https://xgboost.readthedocs.io/en/stable/python/python_api.html)
- [mutagen (ID3-Tagging)](https://mutagen.readthedocs.io/)
