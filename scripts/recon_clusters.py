"""
recon_clusters.py
=================
Reconnaissance-Tool für Chart/Playlist-Cluster.

Mission: Metadaten über Cluster sammeln OHNE Previews/Downloads.
         Entscheidungsgrundlage für Tier-Gewichtung und Genre-Thresholds.

Config: configs/clusters_recon.yaml (SEPARAT von clusters.yaml)
        Enthält: validated, suspicious, excluded Charts mit Status/Metadaten

Was es sammelt:
  - Track-Count pro Playlist
  - Rank-Verteilung (Min/Max/Median/Percentile)
  - Artist-Diversity (Unique Artists / Total Tracks)
  - Release-Date-Verteilung (via Album-API für 15 Sample-Tracks)
  - Overlap zwischen Charts

Was es NICHT macht:
  - Preview-URLs speichern
  - Downloads starten
  - In tracks.jsonl schreiben
  - Last.fm-Lookups

Output:
  - Report:      outputs/reports/recon_TIMESTAMP.json
  - Track-Liste: outputs/recon/tracks_recon_TIMESTAMP.jsonl

Usage:
  python recon_clusters.py                       # Validated + Suspicious Charts
  python recon_clusters.py --scope validated     # Nur validierte Charts
  python recon_clusters.py --scope all           # Alle inkl. excluded
  python recon_clusters.py --charts IT MX CA     # Nur bestimmte Charts
  python recon_clusters.py --dry-run             # Nur zeigen, was getan würde
"""

import sys
import json
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime, timedelta
from collections import Counter
import statistics

import requests

# Füge scripts/ zum Path hinzu für relative Imports
sys.path.insert(0, str(Path(__file__).parent))

from _utils import (
    setup_logging,
    load_paths_config,
    load_clusters_config,
    load_yaml,
    get_charts_config,
    PROJECT_ROOT,
    ensure_dir,
)


# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DEEZER_API_BASE = "https://api.deezer.com"
RECON_CONFIG_PATH = PROJECT_ROOT / "configs" / "clusters_recon.yaml"

# Sample-Tracks für Release-Date-Check (Top 5, Mid 5, Bottom 5)
SAMPLE_SIZE = 5

# Logger
logger = None


# ══════════════════════════════════════════════════════════════════════════════
# DATENSTRUKTUREN
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ReconTrack:
    """Minimales Track-Schema für Recon (ohne Preview-URL)."""
    track_id: int
    title: str
    artist: str
    artist_id: int
    album: str
    album_id: int
    deezer_rank: int
    release_date: Optional[str]  # YYYY-MM-DD oder None
    duration_sec: int
    cluster: str  # Chart-Code (z.B. "IT", "MX")
    position_in_chart: int
    sample_position: Optional[str] = None  # "top", "mid", "bottom" oder None
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChartReconStats:
    """Statistiken für einen Chart-Cluster."""
    chart_code: str
    chart_name: str
    playlist_id: int
    playlist_title: str
    
    # Basis-Metriken
    track_count: int = 0
    unique_artists: int = 0
    artist_diversity: float = 0.0  # unique_artists / track_count
    
    # Rank-Statistiken
    rank_min: Optional[int] = None
    rank_max: Optional[int] = None
    rank_median: Optional[float] = None
    rank_mean: Optional[float] = None
    rank_p25: Optional[float] = None
    rank_p75: Optional[float] = None
    
    # Release-Date-Statistiken (basierend auf 15 Sample-Tracks mit Album-API)
    release_years: dict = field(default_factory=dict)  # {2025: 8, 2024: 5, ...}
    newest_release: Optional[str] = None
    oldest_release: Optional[str] = None
    releases_last_6_months: int = 0
    releases_last_12_months: int = 0
    sample_tracks_with_date: int = 0  # Wie viele der 15 Samples haben ein Datum?
    
    # Top-Artists (zur Spam-Erkennung)
    top_artists: list = field(default_factory=list)  # [(artist, count), ...]
    
    # Sample-Tracks (Top 5 / Mid 5 / Bottom 5 mit Release-Dates)
    sample_tracks_top: list = field(default_factory=list)
    sample_tracks_mid: list = field(default_factory=list)
    sample_tracks_bottom: list = field(default_factory=list)
    
    # Metadaten
    fetched_at: str = ""
    api_errors: int = 0
    album_api_calls: int = 0


@dataclass
class OverlapStats:
    """Overlap zwischen zwei Charts."""
    chart_a: str
    chart_b: str
    shared_track_ids: list = field(default_factory=list)
    shared_count: int = 0
    percent_of_a: float = 0.0
    percent_of_b: float = 0.0
    shared_artists: list = field(default_factory=list)


@dataclass
class ReconReport:
    """Vollständiger Recon-Report."""
    created: str
    charts_analyzed: int = 0
    total_tracks: int = 0
    total_unique_tracks: int = 0
    total_album_api_calls: int = 0
    chart_stats: list = field(default_factory=list)
    overlap_matrix: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    
    # Aggregierte Statistiken
    rank_global_median: Optional[float] = None
    rank_global_p25: Optional[float] = None
    rank_global_p75: Optional[float] = None


# ══════════════════════════════════════════════════════════════════════════════
# API-HELFER
# ══════════════════════════════════════════════════════════════════════════════

def api_get(endpoint: str, params: dict = None, delay: float = 0.25) -> Optional[dict]:
    """Deezer API GET Request mit Rate-Limiting."""
    url = f"{DEEZER_API_BASE}/{endpoint}"
    try:
        time.sleep(delay)
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            if logger:
                logger.warning(f"API Error: {data['error']}")
            return None
        return data
    except Exception as e:
        if logger:
            logger.error(f"API Request failed: {e}")
        return None


def fetch_playlist_tracks(playlist_id: int, limit: int = 200) -> tuple[list[dict], Optional[str]]:
    """
    Holt alle Tracks einer Playlist (mit Pagination).
    
    Returns:
        (tracks_list, error_message)
    """
    all_tracks = []
    url = f"playlist/{playlist_id}/tracks"
    params = {"limit": 100, "index": 0}
    
    while len(all_tracks) < limit:
        data = api_get(url, params)
        if not data:
            return all_tracks, "API error during fetch"
        
        tracks = data.get("data", [])
        if not tracks:
            break
        
        all_tracks.extend(tracks)
        
        # Pagination
        if "next" in data and len(all_tracks) < limit:
            params["index"] += 100
        else:
            break
    
    return all_tracks[:limit], None


def fetch_playlist_info(playlist_id: int) -> Optional[dict]:
    """Holt Playlist-Metadaten."""
    return api_get(f"playlist/{playlist_id}")


def fetch_album_release_date(album_id: int) -> Optional[str]:
    """
    Holt das Release-Date eines Albums via Album-API.
    
    Returns:
        Release-Date als "YYYY-MM-DD" oder None
    """
    data = api_get(f"album/{album_id}")
    if data:
        return data.get("release_date")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# RECON-LOGIK
# ══════════════════════════════════════════════════════════════════════════════

def parse_year_from_date(date_str: Optional[str]) -> Optional[int]:
    """Extrahiert Jahr aus Datum-String (YYYY-MM-DD oder YYYY)."""
    if not date_str:
        return None
    try:
        return int(date_str[:4])
    except (ValueError, IndexError):
        return None


def is_within_months(date_str: Optional[str], months: int) -> bool:
    """Prüft ob ein Datum innerhalb der letzten X Monate liegt."""
    if not date_str:
        return False
    try:
        release = datetime.strptime(date_str[:10], "%Y-%m-%d")
        cutoff = datetime.now() - timedelta(days=months * 30)
        return release >= cutoff
    except (ValueError, IndexError):
        return False


def select_sample_indices(total: int, sample_size: int = SAMPLE_SIZE) -> dict[str, list[int]]:
    """
    Wählt Indizes für Top/Mid/Bottom Samples.
    
    Returns:
        {"top": [0,1,2,3,4], "mid": [47,48,49,50,51], "bottom": [95,96,97,98,99]}
    """
    if total < sample_size * 3:
        # Zu wenige Tracks — alle nehmen
        return {
            "top": list(range(min(sample_size, total))),
            "mid": [],
            "bottom": [],
        }
    
    mid_start = (total - sample_size) // 2
    bottom_start = total - sample_size
    
    return {
        "top": list(range(sample_size)),
        "mid": list(range(mid_start, mid_start + sample_size)),
        "bottom": list(range(bottom_start, total)),
    }


def recon_chart(chart_code: str, chart_config: dict) -> tuple[ChartReconStats, list[ReconTrack]]:
    """
    Führt Reconnaissance für einen Chart durch.
    
    Holt Release-Dates via Album-API nur für 15 Sample-Tracks (Top 5, Mid 5, Bottom 5).
    
    Returns:
        (stats, tracks)
    """
    playlist_id = chart_config.get("playlist_id")
    chart_name = chart_config.get("name", chart_code)
    
    stats = ChartReconStats(
        chart_code=chart_code,
        chart_name=chart_name,
        playlist_id=playlist_id,
        playlist_title="",
        fetched_at=datetime.now().isoformat(),
    )
    tracks = []
    
    # Playlist-Info holen
    info = fetch_playlist_info(playlist_id)
    if not info:
        stats.api_errors += 1
        return stats, tracks
    
    stats.playlist_title = info.get("title", "")
    
    # Tracks holen
    raw_tracks, error = fetch_playlist_tracks(playlist_id)
    if error:
        stats.api_errors += 1
    
    if not raw_tracks:
        return stats, tracks
    
    # Sample-Indizes bestimmen
    sample_indices = select_sample_indices(len(raw_tracks))
    sample_set = set()
    for indices in sample_indices.values():
        sample_set.update(indices)
    
    # Index → Position-Label Mapping
    index_to_position = {}
    for pos, indices in sample_indices.items():
        for idx in indices:
            index_to_position[idx] = pos
    
    # Tracks verarbeiten
    ranks = []
    artists = []
    artist_counter = Counter()
    release_dates = []
    year_counter = Counter()
    
    # Album-ID Cache (um doppelte API-Calls zu vermeiden)
    album_release_cache: dict[int, Optional[str]] = {}
    
    for i, t in enumerate(raw_tracks):
        artist_name = t.get("artist", {}).get("name", "Unknown")
        artist_id = t.get("artist", {}).get("id", 0)
        album = t.get("album", {})
        album_id = album.get("id", 0)
        
        # Release-Date: nur für Sample-Tracks via Album-API holen
        release_date = None
        sample_position = index_to_position.get(i)
        
        if i in sample_set and album_id:
            if album_id in album_release_cache:
                release_date = album_release_cache[album_id]
            else:
                release_date = fetch_album_release_date(album_id)
                album_release_cache[album_id] = release_date
                stats.album_api_calls += 1
        
        recon_track = ReconTrack(
            track_id=t.get("id"),
            title=t.get("title", ""),
            artist=artist_name,
            artist_id=artist_id,
            album=album.get("title", ""),
            album_id=album_id,
            deezer_rank=t.get("rank", 0),
            release_date=release_date,
            duration_sec=t.get("duration", 0),
            cluster=chart_code,
            position_in_chart=i + 1,
            sample_position=sample_position,
        )
        tracks.append(recon_track)
        
        # Statistiken sammeln
        if t.get("rank"):
            ranks.append(t["rank"])
        
        artists.append(artist_name)
        artist_counter[artist_name] += 1
        
        # Release-Dates nur von Samples zählen
        if release_date:
            release_dates.append(release_date)
            year = parse_year_from_date(release_date)
            if year:
                year_counter[year] += 1
    
    # Basis-Metriken
    stats.track_count = len(tracks)
    stats.unique_artists = len(set(artists))
    stats.artist_diversity = stats.unique_artists / stats.track_count if stats.track_count > 0 else 0
    
    # Rank-Statistiken
    if ranks:
        ranks_sorted = sorted(ranks)
        stats.rank_min = ranks_sorted[0]
        stats.rank_max = ranks_sorted[-1]
        stats.rank_median = statistics.median(ranks)
        stats.rank_mean = statistics.mean(ranks)
        n = len(ranks_sorted)
        stats.rank_p25 = ranks_sorted[int(n * 0.25)]
        stats.rank_p75 = ranks_sorted[int(n * 0.75)]
    
    # Release-Date-Statistiken (aus den 15 Samples)
    if release_dates:
        stats.release_years = dict(year_counter.most_common())
        sorted_dates = sorted(release_dates)
        stats.oldest_release = sorted_dates[0]
        stats.newest_release = sorted_dates[-1]
        stats.releases_last_6_months = sum(1 for d in release_dates if is_within_months(d, 6))
        stats.releases_last_12_months = sum(1 for d in release_dates if is_within_months(d, 12))
        stats.sample_tracks_with_date = len(release_dates)
    
    # Top-Artists (für Spam-Erkennung)
    stats.top_artists = artist_counter.most_common(5)
    
    # Sample-Tracks nach Position gruppieren
    for t in tracks:
        sample_info = {
            "artist": t.artist,
            "title": t.title,
            "rank": t.deezer_rank,
            "release_date": t.release_date,
            "position": t.position_in_chart,
        }
        if t.sample_position == "top":
            stats.sample_tracks_top.append(sample_info)
        elif t.sample_position == "mid":
            stats.sample_tracks_mid.append(sample_info)
        elif t.sample_position == "bottom":
            stats.sample_tracks_bottom.append(sample_info)
    
    return stats, tracks


def analyze_overlap(all_tracks: dict[str, list[ReconTrack]]) -> list[OverlapStats]:
    """
    Analysiert Track-Overlap zwischen Charts.
    
    Args:
        all_tracks: {chart_code: [ReconTrack, ...], ...}
    
    Returns:
        Liste von OverlapStats für alle Paare mit Overlap > 0
    """
    chart_codes = list(all_tracks.keys())
    overlaps = []
    
    # Track-IDs pro Chart
    chart_track_ids = {
        code: set(t.track_id for t in tracks)
        for code, tracks in all_tracks.items()
    }
    
    # Artist-Namen pro Chart
    chart_artists = {
        code: set(t.artist for t in tracks)
        for code, tracks in all_tracks.items()
    }
    
    for i, code_a in enumerate(chart_codes):
        for code_b in chart_codes[i+1:]:
            ids_a = chart_track_ids[code_a]
            ids_b = chart_track_ids[code_b]
            shared_ids = ids_a & ids_b
            
            if not shared_ids:
                continue
            
            artists_a = chart_artists[code_a]
            artists_b = chart_artists[code_b]
            shared_artists = artists_a & artists_b
            
            overlap = OverlapStats(
                chart_a=code_a,
                chart_b=code_b,
                shared_track_ids=list(shared_ids),
                shared_count=len(shared_ids),
                percent_of_a=len(shared_ids) / len(ids_a) * 100 if ids_a else 0,
                percent_of_b=len(shared_ids) / len(ids_b) * 100 if ids_b else 0,
                shared_artists=list(shared_artists)[:10],  # Top 10 shared artists
            )
            overlaps.append(overlap)
    
    # Sortieren nach Overlap-Größe
    overlaps.sort(key=lambda x: x.shared_count, reverse=True)
    
    return overlaps


def detect_spam_patterns(stats: ChartReconStats) -> list[str]:
    """
    Erkennt verdächtige Muster in Chart-Daten.
    
    Returns:
        Liste von Warnungen
    """
    warnings = []
    
    # Single-Artist-Dominanz
    if stats.top_artists and stats.track_count > 10:
        top_artist, top_count = stats.top_artists[0]
        dominance = top_count / stats.track_count
        if dominance > 0.3:
            warnings.append(
                f"⚠ {stats.chart_code}: Artist '{top_artist}' dominiert mit {top_count}/{stats.track_count} Tracks ({dominance*100:.0f}%)"
            )
    
    # Niedrige Artist-Diversity
    if stats.artist_diversity < 0.5 and stats.track_count > 20:
        warnings.append(
            f"⚠ {stats.chart_code}: Niedrige Artist-Diversity ({stats.artist_diversity:.2f}) - potentiell manipuliert"
        )
    
    # Alte Releases dominieren (basierend auf 15 Samples)
    if stats.sample_tracks_with_date >= 10:  # Nur warnen wenn genug Samples
        recent_ratio = stats.releases_last_12_months / stats.sample_tracks_with_date
        if recent_ratio < 0.5:
            warnings.append(
                f"⚠ {stats.chart_code}: Nur {stats.releases_last_12_months}/{stats.sample_tracks_with_date} Sample-Tracks aus letzten 12 Monaten - Chart veraltet?"
            )
    
    # Ungewöhnlich hohe/niedrige Ranks
    if stats.rank_median and stats.rank_median > 900000:
        warnings.append(
            f"⚠ {stats.chart_code}: Sehr niedriger Median-Rank ({stats.rank_median:,.0f}) - Nischen-Content?"
        )
    
    return warnings


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def print_chart_stats(stats: ChartReconStats):
    """Druckt Chart-Statistiken auf Console."""
    print(f"\n  {stats.chart_name} ({stats.chart_code})")
    print(f"  {'─' * 50}")
    print(f"  Playlist: {stats.playlist_title} (ID: {stats.playlist_id})")
    print(f"  Tracks: {stats.track_count} | Artists: {stats.unique_artists} | Diversity: {stats.artist_diversity:.2f}")
    
    if stats.rank_median:
        print(f"  Rank: Median={stats.rank_median:,.0f} | P25={stats.rank_p25:,.0f} | P75={stats.rank_p75:,.0f}")
    
    if stats.sample_tracks_with_date:
        print(f"  Releases (Sample): {stats.releases_last_6_months}/{stats.sample_tracks_with_date} (6mo) | {stats.releases_last_12_months}/{stats.sample_tracks_with_date} (12mo)")
        if stats.newest_release:
            print(f"  Date Range: {stats.oldest_release} → {stats.newest_release}")
    
    if stats.top_artists:
        top_str = ", ".join([f"{a[0]} ({a[1]})" for a in stats.top_artists[:3]])
        print(f"  Top Artists: {top_str}")
    
    # Sample-Tracks anzeigen
    if stats.sample_tracks_top:
        print(f"  Top 5:")
        for t in stats.sample_tracks_top[:3]:
            date_str = f" [{t['release_date']}]" if t['release_date'] else ""
            print(f"    #{t['position']:2}: {t['artist']} - {t['title']}{date_str}")


def export_report(report: ReconReport, output_path: Path):
    """Exportiert Report als JSON."""
    data = {
        "created": report.created,
        "summary": {
            "charts_analyzed": report.charts_analyzed,
            "total_tracks": report.total_tracks,
            "total_unique_tracks": report.total_unique_tracks,
            "total_album_api_calls": report.total_album_api_calls,
            "rank_global_median": report.rank_global_median,
            "rank_global_p25": report.rank_global_p25,
            "rank_global_p75": report.rank_global_p75,
        },
        "chart_stats": [asdict(s) for s in report.chart_stats],
        "overlap_matrix": [asdict(o) for o in report.overlap_matrix],
        "warnings": report.warnings,
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Report: {output_path}")


def export_tracks_jsonl(all_tracks: dict[str, list[ReconTrack]], output_path: Path):
    """Exportiert Track-Liste als JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for chart_code, tracks in all_tracks.items():
            for track in tracks:
                f.write(json.dumps(track.to_dict(), ensure_ascii=False) + "\n")
                count += 1
    
    print(f"✓ Tracks: {output_path} ({count} tracks)")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global logger
    
    parser = argparse.ArgumentParser(
        description="Reconnaissance-Tool für Chart/Playlist-Cluster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python recon_clusters.py                       # Validated + Suspicious Charts
  python recon_clusters.py --scope validated     # Nur validierte Charts
  python recon_clusters.py --scope all           # Alle inkl. excluded
  python recon_clusters.py --charts IT MX CA     # Nur bestimmte Charts
  python recon_clusters.py --add-chart IT 1116187241 "Italy"
  python recon_clusters.py --dry-run             # Nur zeigen, was getan würde
        """
    )
    
    parser.add_argument("--scope", choices=["validated", "suspicious", "all", "default"],
                        default="default",
                        help="Welche Chart-Kategorien analysieren (default: validated + suspicious)")
    parser.add_argument("--charts", nargs="+", metavar="CODE",
                        help="Nur diese Charts analysieren (z.B. IT MX CA)")
    parser.add_argument("--add-chart", nargs=3, metavar=("CODE", "PLAYLIST_ID", "NAME"),
                        action="append", dest="extra_charts",
                        help="Zusätzlichen Chart hinzufügen (kann mehrfach verwendet werden)")
    parser.add_argument("--config", type=Path,
                        default=RECON_CONFIG_PATH,
                        help=f"Pfad zur Recon-Config (default: {RECON_CONFIG_PATH})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Nur zeigen, was analysiert würde")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "outputs",
                        help="Output-Verzeichnis (default: outputs/)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Weniger Console-Output")
    
    args = parser.parse_args()
    
    # Logger setup
    logger = setup_logging("recon_clusters")
    
    # Recon-Config laden
    if not args.config.exists():
        print(f"⚠ Config nicht gefunden: {args.config}")
        print("  Erstelle configs/clusters_recon.yaml oder verwende --add-chart")
        
        # Fallback: clusters.yaml verwenden
        clusters_config = load_clusters_config()
        charts_config = get_charts_config(clusters_config)
        
        charts_to_analyze = {}
        if args.charts:
            for code in args.charts:
                code_upper = code.upper()
                if code_upper in charts_config:
                    charts_to_analyze[code_upper] = charts_config[code_upper]
        else:
            charts_to_analyze = charts_config.copy()
    else:
        # Recon-Config verwenden
        recon_config = load_yaml(args.config)
        
        # Charts nach Scope sammeln
        charts_to_analyze = {}
        
        if args.charts:
            # Spezifische Charts - suche in allen Kategorien
            all_charts = {}
            for category in ["existing", "validated", "suspicious", "excluded"]:
                if category in recon_config:
                    all_charts.update(recon_config[category])
            
            for code in args.charts:
                code_upper = code.upper()
                if code_upper in all_charts:
                    charts_to_analyze[code_upper] = all_charts[code_upper]
                else:
                    print(f"⚠ Chart '{code}' nicht in {args.config.name} gefunden")
        else:
            # Nach Scope filtern
            categories_to_include = []
            
            if args.scope == "validated":
                categories_to_include = ["validated"]
            elif args.scope == "suspicious":
                categories_to_include = ["suspicious"]
            elif args.scope == "all":
                categories_to_include = ["existing", "validated", "suspicious", "excluded"]
            else:  # default
                categories_to_include = ["validated", "suspicious"]
            
            for category in categories_to_include:
                if category in recon_config:
                    for code, config in recon_config[category].items():
                        # Überspringe Charts ohne playlist_id (z.B. "existing" oder "excluded" ohne ID)
                        if config.get("playlist_id") or category == "existing":
                            charts_to_analyze[code] = config
    
    # Extra-Charts hinzufügen
    if args.extra_charts:
        for code, playlist_id, name in args.extra_charts:
            charts_to_analyze[code.upper()] = {
                "name": name,
                "playlist_id": int(playlist_id),
            }
    
    # Charts ohne playlist_id filtern (für Recon brauchen wir die ID)
    charts_to_analyze = {
        code: config for code, config in charts_to_analyze.items()
        if config.get("playlist_id")
    }
    
    if not charts_to_analyze:
        print("Keine Charts zum Analysieren gefunden.")
        print("Verwende --add-chart CODE PLAYLIST_ID NAME oder konfiguriere Charts in clusters_recon.yaml")
        sys.exit(1)
    
    # Dry-Run?
    if args.dry_run:
        print(f"\n{'='*60}")
        print("DRY RUN - Würde folgende Charts analysieren:")
        print(f"{'='*60}\n")
        for code, config in sorted(charts_to_analyze.items()):
            status = config.get("status", "?")
            name = config.get("name", "?")
            playlist_id = config.get("playlist_id", "?")
            print(f"  [{status:10}] {code}: {name} (ID: {playlist_id})")
        print(f"\n  Total: {len(charts_to_analyze)} Charts")
        print(f"  Geschätzte Album-API-Calls: ~{len(charts_to_analyze) * 15} (15 pro Chart)")
        sys.exit(0)
    
    # === RECON STARTEN ===
    
    print(f"\n{'='*60}")
    print(f"RECON CLUSTERS - {len(charts_to_analyze)} Charts")
    print(f"(~{len(charts_to_analyze) * 15} Album-API-Calls für Release-Dates)")
    print(f"{'='*60}")
    
    report = ReconReport(created=datetime.now().isoformat())
    all_tracks: dict[str, list[ReconTrack]] = {}
    all_ranks = []
    
    for code, config in charts_to_analyze.items():
        print(f"\n  Analysiere {config.get('name', code)} ({code})...", end="", flush=True)
        
        stats, tracks = recon_chart(code, config)
        all_tracks[code] = tracks
        report.chart_stats.append(stats)
        report.total_album_api_calls += stats.album_api_calls
        
        if stats.api_errors:
            print(f" ⚠ API Errors: {stats.api_errors}")
        else:
            print(f" ✓ {stats.track_count} tracks, {stats.album_api_calls} album calls")
        
        # Ranks sammeln für globale Statistiken
        for t in tracks:
            if t.deezer_rank:
                all_ranks.append(t.deezer_rank)
        
        # Spam-Erkennung
        warnings = detect_spam_patterns(stats)
        report.warnings.extend(warnings)
        
        if not args.quiet:
            print_chart_stats(stats)
    
    # Overlap-Analyse
    print(f"\n{'='*60}")
    print("OVERLAP ANALYSIS")
    print(f"{'='*60}")
    
    report.overlap_matrix = analyze_overlap(all_tracks)
    
    if report.overlap_matrix:
        print(f"\n  {'Chart A':10} {'Chart B':10} {'Shared':>8} {'%A':>8} {'%B':>8}")
        print(f"  {'-'*50}")
        for o in report.overlap_matrix[:15]:  # Top 15
            print(f"  {o.chart_a:10} {o.chart_b:10} {o.shared_count:8} {o.percent_of_a:7.1f}% {o.percent_of_b:7.1f}%")
    else:
        print("  Kein signifikanter Overlap gefunden.")
    
    # Globale Statistiken
    report.charts_analyzed = len(charts_to_analyze)
    report.total_tracks = sum(len(t) for t in all_tracks.values())
    report.total_unique_tracks = len(set(t.track_id for tracks in all_tracks.values() for t in tracks))
    
    if all_ranks:
        sorted_ranks = sorted(all_ranks)
        n = len(sorted_ranks)
        report.rank_global_median = statistics.median(all_ranks)
        report.rank_global_p25 = sorted_ranks[int(n * 0.25)]
        report.rank_global_p75 = sorted_ranks[int(n * 0.75)]
    
    # Warnings
    if report.warnings:
        print(f"\n{'='*60}")
        print("WARNINGS")
        print(f"{'='*60}\n")
        for w in report.warnings:
            print(f"  {w}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")
    print(f"  Charts analyzed:    {report.charts_analyzed}")
    print(f"  Total tracks:       {report.total_tracks}")
    print(f"  Unique tracks:      {report.total_unique_tracks}")
    print(f"  Album API calls:    {report.total_album_api_calls}")
    if report.rank_global_median:
        print(f"  Global Rank:        Median={report.rank_global_median:,.0f} | P25={report.rank_global_p25:,.0f} | P75={report.rank_global_p75:,.0f}")
    
    # Export
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report_path = args.output_dir / "reports" / f"recon_{timestamp}.json"
    export_report(report, report_path)
    
    tracks_path = args.output_dir / "recon" / f"tracks_recon_{timestamp}.jsonl"
    export_tracks_jsonl(all_tracks, tracks_path)
    
    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
