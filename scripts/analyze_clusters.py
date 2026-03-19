"""
analyze_clusters.py
===================
Multi-Purpose Analyse-Tool für SpotilyzerTraining.

Use Cases:
  1. Chart-Discovery: Deezer nach Chart-Playlists für weitere Länder abfragen
  2. Cluster-Statistiken: Verteilungen, Overlap, Rank-Statistiken
  3. Sanity-Checks: Vor/nach Scouting-Läufen, API-Verfügbarkeit
  4. Debugging: Fehlende Labels, Previews, Last.fm-Matches

Subcommands:
  --discover-charts     Deezer nach Chart-Playlists für Länderliste abfragen
  --cluster-stats       Statistiken pro Cluster aus tracks.jsonl
  --label-distribution  Hit/Mid/Flop-Verteilung pro Cluster
  --overlap             Track-Overlap zwischen Clustern
  --sanity              Vor-Scouting-Check (Playlist-IDs, API-Verfügbarkeit)
  --full                Alles zusammen als Report

Output:
  JSON Report wird automatisch gespeichert in outputs/reports/
  --no-save        Deaktiviert Auto-Save
  --output FILE    Eigener Output-Pfad (*.json oder *.md)
"""

import sys
import json
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import defaultdict
from datetime import datetime

import requests

# Füge scripts/ zum Path hinzu für relative Imports
sys.path.insert(0, str(Path(__file__).parent))

from _utils import (
    setup_logging,
    load_paths_config,
    load_clusters_config,
    get_genre_clusters,
    get_charts_config,
    PROJECT_ROOT,
)
from utils.metadata import read_tracks, get_tracks_jsonl_path

# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DEEZER_API_BASE = "https://api.deezer.com"

# Länder, für die wir Chart-Playlists suchen wollen
# (zusätzlich zu den bereits konfigurierten)
CHART_DISCOVERY_COUNTRIES = [
    ("IT", "Italy"),
    ("MX", "Mexico"),
    ("CA", "Canada"),
    ("AU", "Australia"),
    ("PL", "Poland"),
    ("NL", "Netherlands"),
    ("SE", "Sweden"),
    ("KR", "South Korea"),
    ("AT", "Austria"),
    ("CH", "Switzerland"),
    ("BE", "Belgium"),
    ("NO", "Norway"),
    ("DK", "Denmark"),
    ("FI", "Finland"),
    ("PT", "Portugal"),
    ("AR", "Argentina"),
    ("CO", "Colombia"),
    ("CL", "Chile"),
    ("ZA", "South Africa"),
    ("IN", "India"),
    ("ID", "Indonesia"),
    ("PH", "Philippines"),
    ("TH", "Thailand"),
    ("TR", "Turkey"),
    ("EG", "Egypt"),
    ("SA", "Saudi Arabia"),
    ("AE", "UAE"),
    ("NZ", "New Zealand"),
    ("IE", "Ireland"),
    ("SG", "Singapore"),
    ("MY", "Malaysia"),
]

# Logger
logger = None


# ══════════════════════════════════════════════════════════════════════════════
# DATENSTRUKTUREN
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ChartInfo:
    """Info über eine entdeckte Chart-Playlist."""
    country_code: str
    country_name: str
    playlist_id: Optional[int] = None
    playlist_title: Optional[str] = None
    track_count: int = 0
    found: bool = False
    error: Optional[str] = None
    # Zusätzliche Infos zur Bewertung der Playlist-Qualität
    creator_name: Optional[str] = None
    creator_id: Optional[int] = None
    is_official_deezer: bool = False
    followers: Optional[int] = None
    last_updated: Optional[str] = None
    # Stichprobe der ersten 3 Tracks (zur Aktualitäts-Prüfung)
    sample_tracks: list = field(default_factory=list)


@dataclass
class ClusterStats:
    """Statistiken für einen Cluster."""
    cluster_id: str
    display_name: str
    track_count: int = 0
    hit_count: int = 0
    mid_count: int = 0
    flop_count: int = 0
    unlabeled_count: int = 0
    validated_count: int = 0
    contested_count: int = 0
    single_source_count: int = 0
    rank_min: Optional[int] = None
    rank_max: Optional[int] = None
    rank_median: Optional[float] = None
    rank_p25: Optional[float] = None
    rank_p75: Optional[float] = None
    has_preview_count: int = 0
    missing_preview_count: int = 0
    lastfm_matched_count: int = 0
    lastfm_missing_count: int = 0


@dataclass
class OverlapInfo:
    """Overlap zwischen zwei Clustern."""
    cluster_a: str
    cluster_b: str
    shared_tracks: int
    percent_of_a: float
    percent_of_b: float


@dataclass
class AnalysisReport:
    """Vollständiger Analyse-Report."""
    created: str
    charts_configured: dict = field(default_factory=dict)
    charts_discovered: list = field(default_factory=list)
    cluster_stats: list = field(default_factory=list)
    overlap_matrix: list = field(default_factory=list)
    totals: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)


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


def get_playlist_info(playlist_id: int) -> Optional[dict]:
    """Holt Playlist-Metadaten von Deezer."""
    return api_get(f"playlist/{playlist_id}")


def get_playlist_details(playlist_id: int) -> Optional[dict]:
    """
    Holt detaillierte Infos zu einer Playlist.
    Inklusive Follower-Count und Sample-Tracks.
    """
    data = api_get(f"playlist/{playlist_id}")
    if not data:
        return None
    
    # Extrahiere Sample-Tracks (erste 3)
    sample_tracks = []
    tracks_data = data.get("tracks", {}).get("data", [])
    for t in tracks_data[:3]:
        sample_tracks.append({
            "artist": t.get("artist", {}).get("name", "?"),
            "title": t.get("title", "?"),
        })
    
    return {
        "id": data.get("id"),
        "title": data.get("title"),
        "nb_tracks": data.get("nb_tracks", 0),
        "fans": data.get("fans", 0),  # Follower
        "creator_id": data.get("creator", {}).get("id"),
        "creator_name": data.get("creator", {}).get("name"),
        "sample_tracks": sample_tracks,
    }


def search_chart_playlist(country_code: str, country_name: str) -> ChartInfo:
    """
    Sucht nach einer Chart-Playlist für ein Land.
    
    Strategie:
    1. Suche nach "Top {country_name}" Playlist
    2. Suche nach "Top {country_code}" Playlist
    3. Suche nach "{country_name} Chart" Playlist
    
    Sammelt zusätzliche Infos:
    - Ersteller (offiziell vs. User-kuratiert)
    - Follower-Anzahl (Indikator für Verbreitung)
    - Track-Anzahl
    """
    info = ChartInfo(country_code=country_code, country_name=country_name)
    
    # Deezer Editorial User-IDs
    # 2 = Deezer (original Editorial)
    # 637006841 = "Deezer Charts" (automatische Chart-Playlists pro Land)
    # 2748989402 = Deezer Editorial (neuerer Account)
    # 4036701362 = "Deezer Best Of" (Best-Of-Compilations)
    # 4260225282 = weitere Editorial-Variante
    DEEZER_EDITORIAL_IDS = {2, 637006841, 2748989402, 4036701362, 4260225282}
    
    search_terms = [
        f"Top {country_name}",
        f"Top {country_code}",
        f"{country_name} Chart",
        f"{country_name} Top 50",
        f"{country_name} Top 100",
    ]
    
    best_unofficial = None  # Beste inoffizielle Option speichern
    
    for term in search_terms:
        data = api_get("search/playlist", params={"q": term, "limit": 15})
        if not data or "data" not in data:
            continue
        
        for playlist in data["data"]:
            title = playlist.get("title", "").lower()
            creator = playlist.get("user", {})
            creator_id = creator.get("id")
            
            # Prüfe ob es eine Chart-Playlist sein könnte
            is_match = (
                ("top" in title and country_name.lower() in title) or
                ("chart" in title and country_name.lower() in title) or
                (f"top {country_code.lower()}" in title) or
                (f"{country_name.lower()} hits" in title)
            )
            
            if not is_match:
                continue
            
            # Offizielle Deezer-Playlist?
            if creator_id in DEEZER_EDITORIAL_IDS:
                info.playlist_id = playlist["id"]
                info.playlist_title = playlist["title"]
                info.track_count = playlist.get("nb_tracks", 0)
                info.creator_name = creator.get("name", "Deezer")
                info.creator_id = creator_id
                info.is_official_deezer = True
                
                # Hole echte Follower-Zahl und Sample-Tracks via direktem API-Call
                details = get_playlist_details(playlist["id"])
                if details:
                    info.followers = details.get("fans", 0)
                    info.sample_tracks = details.get("sample_tracks", [])
                else:
                    info.followers = playlist.get("fans", 0)
                
                info.found = True
                return info
            
            # Inoffizielle Playlist - speichere die beste (meiste Follower)
            fans = playlist.get("fans", 0)
            tracks = playlist.get("nb_tracks", 0)
            
            if tracks >= 30:  # Mindestens 30 Tracks
                if best_unofficial is None or fans > best_unofficial.get("fans", 0):
                    best_unofficial = {
                        "playlist_id": playlist["id"],
                        "title": playlist["title"],
                        "nb_tracks": tracks,
                        "fans": fans,
                        "creator_name": creator.get("name"),
                        "creator_id": creator_id,
                    }
    
    # Keine offizielle gefunden - nimm beste inoffizielle
    if best_unofficial:
        info.playlist_id = best_unofficial["playlist_id"]
        info.playlist_title = best_unofficial["title"]
        info.track_count = best_unofficial["nb_tracks"]
        info.creator_name = best_unofficial["creator_name"]
        info.creator_id = best_unofficial["creator_id"]
        info.is_official_deezer = False
        
        # Hole echte Follower-Zahl und Sample-Tracks via direktem API-Call
        # (Search-API liefert oft 0, obwohl Playlist Follower hat)
        details = get_playlist_details(best_unofficial["playlist_id"])
        if details:
            info.followers = details.get("fans", 0)
            info.sample_tracks = details.get("sample_tracks", [])
        else:
            info.followers = best_unofficial["fans"]
        
        info.found = True
        info.error = f"User-kuratiert von '{best_unofficial['creator_name']}' ({info.followers:,} Follower)"
        return info
    
    info.error = "No chart playlist found"
    return info


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSE-FUNKTIONEN
# ══════════════════════════════════════════════════════════════════════════════

def discover_charts(existing_charts: dict) -> list[ChartInfo]:
    """
    Entdeckt Chart-Playlists für Länder, die noch nicht konfiguriert sind.
    
    Args:
        existing_charts: Bereits konfigurierte Charts aus clusters.yaml
    
    Returns:
        Liste von ChartInfo für entdeckte Charts
    """
    existing_codes = set(existing_charts.keys())
    discovered = []
    
    print(f"\n{'='*60}")
    print("CHART DISCOVERY")
    print(f"{'='*60}")
    print(f"Bereits konfiguriert: {', '.join(sorted(existing_codes))}")
    print(f"Suche nach: {len(CHART_DISCOVERY_COUNTRIES)} weiteren Ländern\n")
    
    for code, name in CHART_DISCOVERY_COUNTRIES:
        if code in existing_codes:
            continue
        
        print(f"  Suche {name} ({code})...", end=" ", flush=True)
        info = search_chart_playlist(code, name)
        discovered.append(info)
        
        if info.found:
            official_marker = "★" if info.is_official_deezer else "☆"
            print(f"{official_marker} {info.playlist_title}")
            print(f"      ID: {info.playlist_id} | {info.track_count} tracks | {info.followers or 0:,} followers")
            print(f"      Ersteller: {info.creator_name} (ID: {info.creator_id})")
            if info.sample_tracks:
                sample_str = ", ".join([f"{t['artist']} - {t['title']}" for t in info.sample_tracks[:2]])
                print(f"      Sample: {sample_str}")
            if info.error and not info.is_official_deezer:
                print(f"      ⚠ {info.error}")
        else:
            print(f"✗ {info.error}")
    
    return discovered


def validate_existing_charts(charts: dict) -> list[dict]:
    """Validiert, dass konfigurierte Chart-Playlists noch existieren."""
    results = []
    
    print(f"\n{'='*60}")
    print("CHART VALIDATION")
    print(f"{'='*60}\n")
    
    for code, config in charts.items():
        playlist_id = config.get("playlist_id")
        name = config.get("name", code)
        
        print(f"  Prüfe {name} ({code}, ID: {playlist_id})...", end=" ", flush=True)
        
        data = get_playlist_info(playlist_id)
        if data:
            track_count = data.get("nb_tracks", 0)
            title = data.get("title", "")
            print(f"✓ {title} ({track_count} tracks)")
            results.append({
                "code": code,
                "name": name,
                "playlist_id": playlist_id,
                "valid": True,
                "track_count": track_count,
                "title": title,
            })
        else:
            print("✗ Playlist nicht gefunden!")
            results.append({
                "code": code,
                "name": name,
                "playlist_id": playlist_id,
                "valid": False,
                "error": "Playlist not found or API error",
            })
    
    return results


def calculate_cluster_stats(tracks: list[dict], cluster_id: str, display_name: str) -> ClusterStats:
    """Berechnet Statistiken für einen Cluster."""
    stats = ClusterStats(cluster_id=cluster_id, display_name=display_name)
    
    cluster_tracks = [t for t in tracks if cluster_id in t.get("clusters", [])]
    stats.track_count = len(cluster_tracks)
    
    if not cluster_tracks:
        return stats
    
    # Label-Verteilung
    for track in cluster_tracks:
        label = track.get("label", "")
        if label == "hit":
            stats.hit_count += 1
        elif label == "mid":
            stats.mid_count += 1
        elif label == "flop":
            stats.flop_count += 1
        else:
            stats.unlabeled_count += 1
        
        # Robustness
        robustness = track.get("robustness", "")
        if robustness == "validated":
            stats.validated_count += 1
        elif robustness == "contested":
            stats.contested_count += 1
        elif robustness == "single_source":
            stats.single_source_count += 1
        
        # Preview
        if track.get("file_path"):
            stats.has_preview_count += 1
        else:
            stats.missing_preview_count += 1
        
        # Last.fm
        if track.get("lastfm_playcount"):
            stats.lastfm_matched_count += 1
        else:
            stats.lastfm_missing_count += 1
    
    # Rank-Statistiken
    ranks = [t.get("deezer_rank") for t in cluster_tracks if t.get("deezer_rank")]
    if ranks:
        ranks_sorted = sorted(ranks)
        stats.rank_min = ranks_sorted[0]
        stats.rank_max = ranks_sorted[-1]
        n = len(ranks_sorted)
        stats.rank_median = ranks_sorted[n // 2] if n % 2 == 1 else (ranks_sorted[n//2 - 1] + ranks_sorted[n//2]) / 2
        stats.rank_p25 = ranks_sorted[int(n * 0.25)]
        stats.rank_p75 = ranks_sorted[int(n * 0.75)]
    
    return stats


def analyze_clusters(tracks: list[dict], clusters_config: dict) -> list[ClusterStats]:
    """Analysiert alle Cluster."""
    genre_clusters = get_genre_clusters(clusters_config)
    charts = get_charts_config(clusters_config)
    
    all_stats = []
    
    print(f"\n{'='*60}")
    print("CLUSTER STATISTICS")
    print(f"{'='*60}\n")
    
    # Genre-Cluster
    print("Genre-Cluster:")
    print("-" * 40)
    for cluster_id, config in genre_clusters.items():
        display_name = config.get("display_name", cluster_id)
        stats = calculate_cluster_stats(tracks, cluster_id, display_name)
        all_stats.append(stats)
        
        hit_pct = (stats.hit_count / stats.track_count * 100) if stats.track_count > 0 else 0
        print(f"  {display_name:30} {stats.track_count:5} tracks | "
              f"H:{stats.hit_count:4} M:{stats.mid_count:4} F:{stats.flop_count:4} | "
              f"Hit%: {hit_pct:5.1f}%")
    
    # Chart-Cluster
    if charts:
        print("\nChart-Cluster:")
        print("-" * 40)
        for code, config in charts.items():
            cluster_id = f"charts_{code.lower()}"
            display_name = f"Charts {config.get('name', code)}"
            stats = calculate_cluster_stats(tracks, cluster_id, display_name)
            all_stats.append(stats)
            
            hit_pct = (stats.hit_count / stats.track_count * 100) if stats.track_count > 0 else 0
            print(f"  {display_name:30} {stats.track_count:5} tracks | "
                  f"H:{stats.hit_count:4} M:{stats.mid_count:4} F:{stats.flop_count:4} | "
                  f"Hit%: {hit_pct:5.1f}%")
    
    return all_stats


def analyze_overlap(tracks: list[dict], clusters_config: dict) -> list[OverlapInfo]:
    """Analysiert Track-Overlap zwischen Clustern."""
    genre_clusters = get_genre_clusters(clusters_config)
    cluster_ids = list(genre_clusters.keys())
    
    # Tracks pro Cluster sammeln
    cluster_tracks = {}
    for cid in cluster_ids:
        cluster_tracks[cid] = set(
            t["track_id"] for t in tracks if cid in t.get("clusters", [])
        )
    
    overlaps = []
    
    print(f"\n{'='*60}")
    print("CLUSTER OVERLAP (>5% shared)")
    print(f"{'='*60}\n")
    
    for i, cid_a in enumerate(cluster_ids):
        for cid_b in cluster_ids[i+1:]:
            tracks_a = cluster_tracks[cid_a]
            tracks_b = cluster_tracks[cid_b]
            shared = tracks_a & tracks_b
            
            if not shared:
                continue
            
            pct_a = len(shared) / len(tracks_a) * 100 if tracks_a else 0
            pct_b = len(shared) / len(tracks_b) * 100 if tracks_b else 0
            
            if pct_a > 5 or pct_b > 5:
                info = OverlapInfo(
                    cluster_a=cid_a,
                    cluster_b=cid_b,
                    shared_tracks=len(shared),
                    percent_of_a=pct_a,
                    percent_of_b=pct_b,
                )
                overlaps.append(info)
                
                name_a = genre_clusters[cid_a].get("display_name", cid_a)
                name_b = genre_clusters[cid_b].get("display_name", cid_b)
                print(f"  {name_a:25} ↔ {name_b:25}: "
                      f"{len(shared):4} tracks ({pct_a:5.1f}% / {pct_b:5.1f}%)")
    
    if not overlaps:
        print("  Kein signifikanter Overlap gefunden.")
    
    return overlaps


def find_issues(tracks: list[dict]) -> list[str]:
    """Findet potenzielle Probleme im Dataset."""
    issues = []
    
    print(f"\n{'='*60}")
    print("ISSUES & WARNINGS")
    print(f"{'='*60}\n")
    
    # Tracks ohne Label
    unlabeled = [t for t in tracks if not t.get("label")]
    if unlabeled:
        msg = f"⚠ {len(unlabeled)} tracks ohne Label"
        issues.append(msg)
        print(f"  {msg}")
    
    # Tracks ohne Preview
    no_preview = [t for t in tracks if not t.get("file_path")]
    if no_preview:
        msg = f"⚠ {len(no_preview)} tracks ohne Preview-Datei"
        issues.append(msg)
        print(f"  {msg}")
    
    # Tracks ohne Last.fm-Match
    no_lastfm = [t for t in tracks if not t.get("lastfm_playcount") and t.get("label")]
    if no_lastfm:
        msg = f"ℹ {len(no_lastfm)} gelabelte tracks ohne Last.fm-Daten"
        issues.append(msg)
        print(f"  {msg}")
    
    # Contested Labels
    contested = [t for t in tracks if t.get("robustness") == "contested"]
    if contested:
        contested_pct = len(contested) / len(tracks) * 100 if tracks else 0
        msg = f"ℹ {len(contested)} tracks ({contested_pct:.1f}%) mit 'contested' Robustheit"
        issues.append(msg)
        print(f"  {msg}")
    
    # Label-Imbalance
    labels = [t.get("label") for t in tracks if t.get("label")]
    if labels:
        from collections import Counter
        counts = Counter(labels)
        total = len(labels)
        for label, count in counts.items():
            pct = count / total * 100
            if pct < 10:
                msg = f"⚠ Klassen-Imbalance: '{label}' nur {pct:.1f}% ({count} samples)"
                issues.append(msg)
                print(f"  {msg}")
    
    if not issues:
        print("  ✓ Keine kritischen Issues gefunden.")
    
    return issues


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT-FORMATIERUNG
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(report: AnalysisReport):
    """Druckt eine Zusammenfassung des Reports."""
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")
    
    totals = report.totals
    print(f"  Total Tracks:     {totals.get('total_tracks', 0):,}")
    print(f"  Mit Label:        {totals.get('labeled_tracks', 0):,}")
    print(f"  Mit Preview:      {totals.get('with_preview', 0):,}")
    print(f"  Mit Last.fm:      {totals.get('with_lastfm', 0):,}")
    print()
    print(f"  Hits:             {totals.get('hits', 0):,} ({totals.get('hit_pct', 0):.1f}%)")
    print(f"  Mids:             {totals.get('mids', 0):,} ({totals.get('mid_pct', 0):.1f}%)")
    print(f"  Flops:            {totals.get('flops', 0):,} ({totals.get('flop_pct', 0):.1f}%)")
    print()
    print(f"  Genre-Cluster:    {totals.get('genre_clusters', 0)}")
    print(f"  Chart-Cluster:    {totals.get('chart_clusters', 0)}")
    
    if report.charts_discovered:
        found = [c for c in report.charts_discovered if c.found]
        print(f"\n  Neue Charts entdeckt: {len(found)}")


def export_json(report: AnalysisReport, output_path: Path):
    """Exportiert Report als JSON."""
    # Dataclasses zu dicts konvertieren
    data = {
        "created": report.created,
        "charts_configured": report.charts_configured,
        "charts_discovered": [asdict(c) for c in report.charts_discovered],
        "cluster_stats": [asdict(s) for s in report.cluster_stats],
        "overlap_matrix": [asdict(o) for o in report.overlap_matrix],
        "totals": report.totals,
        "issues": report.issues,
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Report exportiert nach: {output_path}")


def export_markdown(report: AnalysisReport, output_path: Path):
    """Exportiert Report als Markdown."""
    lines = [
        f"# Spotilyzer Cluster Analysis Report",
        f"",
        f"**Generated:** {report.created}",
        f"",
        f"## Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
    ]
    
    for key, value in report.totals.items():
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.1f}% |")
        else:
            lines.append(f"| {key} | {value:,} |")
    
    if report.charts_discovered:
        found = [c for c in report.charts_discovered if c.found]
        lines.extend([
            f"",
            f"## Discovered Charts ({len(found)} found)",
            f"",
            f"| Country | Code | Playlist ID | Title | Tracks |",
            f"|---------|------|-------------|-------|--------|",
        ])
        for chart in report.charts_discovered:
            if chart.found:
                lines.append(
                    f"| {chart.country_name} | {chart.country_code} | "
                    f"{chart.playlist_id} | {chart.playlist_title} | {chart.track_count} |"
                )
    
    if report.cluster_stats:
        lines.extend([
            f"",
            f"## Cluster Statistics",
            f"",
            f"| Cluster | Tracks | Hits | Mids | Flops | Hit% |",
            f"|---------|--------|------|------|-------|------|",
        ])
        for stats in report.cluster_stats:
            hit_pct = (stats.hit_count / stats.track_count * 100) if stats.track_count > 0 else 0
            lines.append(
                f"| {stats.display_name} | {stats.track_count} | "
                f"{stats.hit_count} | {stats.mid_count} | {stats.flop_count} | {hit_pct:.1f}% |"
            )
    
    if report.issues:
        lines.extend([
            f"",
            f"## Issues",
            f"",
        ])
        for issue in report.issues:
            lines.append(f"- {issue}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"\n✓ Report exportiert nach: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global logger
    
    parser = argparse.ArgumentParser(
        description="Cluster-Analyse-Tool für SpotilyzerTraining",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python analyze_clusters.py --discover-charts
  python analyze_clusters.py --cluster-stats
  python analyze_clusters.py --full --output report.json
  python analyze_clusters.py --sanity
        """
    )
    
    # Subcommands als Flags
    parser.add_argument("--discover-charts", action="store_true",
                        help="Suche nach Chart-Playlists für weitere Länder")
    parser.add_argument("--cluster-stats", action="store_true",
                        help="Zeige Statistiken pro Cluster")
    parser.add_argument("--label-distribution", action="store_true",
                        help="Zeige Hit/Mid/Flop-Verteilung pro Cluster")
    parser.add_argument("--overlap", action="store_true",
                        help="Analysiere Track-Overlap zwischen Clustern")
    parser.add_argument("--sanity", action="store_true",
                        help="Sanity-Check: Validiere konfigurierte Charts")
    parser.add_argument("--full", action="store_true",
                        help="Vollständiger Report (alle Analysen)")
    
    # Output
    parser.add_argument("--output", "-o", type=Path,
                        help="Output-Datei (*.json oder *.md). Default: outputs/reports/cluster_analysis_TIMESTAMP.json")
    parser.add_argument("--no-save", action="store_true",
                        help="Keinen Report speichern (nur Console-Output)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Nur Fehler ausgeben")
    
    args = parser.parse_args()
    
    # Mindestens ein Subcommand erforderlich
    if not any([args.discover_charts, args.cluster_stats, args.label_distribution,
                args.overlap, args.sanity, args.full]):
        parser.print_help()
        print("\n⚠ Mindestens ein Analyse-Flag erforderlich.")
        sys.exit(1)
    
    # Logger setup
    logger = setup_logging("analyze_clusters")
    
    # Configs laden
    paths_config = load_paths_config()
    clusters_config = load_clusters_config()
    charts_config = get_charts_config(clusters_config)
    
    # Tracks laden (falls benötigt)
    tracks = []
    needs_tracks = args.cluster_stats or args.label_distribution or args.overlap or args.full
    if needs_tracks:
        tracks_file = get_tracks_jsonl_path(paths_config)
        if tracks_file.exists():
            tracks = list(read_tracks(tracks_file))
            print(f"Geladen: {len(tracks)} tracks aus {tracks_file}")
        else:
            print(f"⚠ Keine tracks.jsonl gefunden: {tracks_file}")
    
    # Report initialisieren
    report = AnalysisReport(
        created=datetime.now().isoformat(),
        charts_configured={code: cfg.get("name", code) for code, cfg in charts_config.items()},
    )
    
    # Analysen durchführen
    if args.sanity or args.full:
        validation = validate_existing_charts(charts_config)
        # Füge zu Report hinzu
        for v in validation:
            if not v.get("valid"):
                report.issues.append(f"Invalid chart: {v['code']} - {v.get('error')}")
    
    if args.discover_charts or args.full:
        report.charts_discovered = discover_charts(charts_config)
    
    if args.cluster_stats or args.label_distribution or args.full:
        report.cluster_stats = analyze_clusters(tracks, clusters_config)
    
    if args.overlap or args.full:
        report.overlap_matrix = analyze_overlap(tracks, clusters_config)
    
    # Issues finden
    if tracks:
        report.issues.extend(find_issues(tracks))
    
    # Totals berechnen
    if tracks:
        labeled = [t for t in tracks if t.get("label")]
        hits = [t for t in tracks if t.get("label") == "hit"]
        mids = [t for t in tracks if t.get("label") == "mid"]
        flops = [t for t in tracks if t.get("label") == "flop"]
        
        report.totals = {
            "total_tracks": len(tracks),
            "labeled_tracks": len(labeled),
            "with_preview": len([t for t in tracks if t.get("file_path")]),
            "with_lastfm": len([t for t in tracks if t.get("lastfm_playcount")]),
            "hits": len(hits),
            "mids": len(mids),
            "flops": len(flops),
            "hit_pct": len(hits) / len(labeled) * 100 if labeled else 0,
            "mid_pct": len(mids) / len(labeled) * 100 if labeled else 0,
            "flop_pct": len(flops) / len(labeled) * 100 if labeled else 0,
            "genre_clusters": len(get_genre_clusters(clusters_config)),
            "chart_clusters": len(charts_config),
        }
    
    # Summary drucken
    if not args.quiet:
        print_summary(report)
    
    # Export - Default: JSON speichern
    if not args.no_save:
        if args.output:
            output_path = args.output
        else:
            # Default: outputs/reports/cluster_analysis_TIMESTAMP.json
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            reports_dir = PROJECT_ROOT / "outputs" / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            output_path = reports_dir / f"cluster_analysis_{timestamp}.json"
        
        if output_path.suffix == ".json":
            export_json(report, output_path)
        elif output_path.suffix == ".md":
            export_markdown(report, output_path)
        else:
            print(f"⚠ Unbekanntes Output-Format: {output_path.suffix}")
            print("  Unterstützt: .json, .md")
    
    # Entdeckte Charts als YAML-Snippet ausgeben
    if args.discover_charts or args.full:
        found_charts = [c for c in report.charts_discovered if c.found]
        if found_charts:
            print(f"\n{'='*60}")
            print("YAML-SNIPPET FÜR clusters.yaml")
            print(f"{'='*60}\n")
            print("# Kopiere folgende Zeilen in den 'charts:' Block:\n")
            for chart in found_charts:
                print(f"  {chart.country_code}:")
                print(f'    name: "{chart.country_name}"')
                print(f"    playlist_id: {chart.playlist_id}")
                print()


if __name__ == "__main__":
    main()
