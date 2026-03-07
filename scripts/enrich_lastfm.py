"""
enrich_lastfm.py
================
Reichert Deezer-Scouting-Daten mit Last.fm-Metriken an.

Workflow:
1. Liest scouted_tracks.csv (Deezer-Daten)
2. Sucht jeden Track auf Last.fm via Artist + Title
3. Fuzzy-Matching für robuste Zuordnung
4. Speichert playcount, listeners, match_confidence

Voraussetzungen:
- Last.fm API-Key in .env (LASTFM_API_KEY)
- pip install pylast rapidfuzz python-dotenv tqdm pandas

Autor: Claude (für Andreas Vogelsang / Spotilyzer)
Datum: 2026-03-07
"""

import os
import sys
import re
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

try:
    import pylast
except ImportError:
    print("pylast nicht installiert. Bitte: pip install pylast")
    sys.exit(1)

try:
    from rapidfuzz import fuzz
except ImportError:
    print("rapidfuzz nicht installiert. Bitte: pip install rapidfuzz")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

REQUEST_DELAY = 0.25  # Sekunden zwischen API-Calls
MATCH_THRESHOLD = 85  # Minimum Fuzzy-Score für Match (0-100)
MAX_RETRIES = 3


# ══════════════════════════════════════════════════════════════════════════════
# HILFSFUNKTIONEN
# ══════════════════════════════════════════════════════════════════════════════

def normalize_string(s: str) -> str:
    """Normalisiert String für Fuzzy-Matching."""
    if not s:
        return ""
    s = s.lower()
    # Entferne Klammern und deren Inhalt (feat., remix, live, etc.)
    s = re.sub(r"\s*[\(\[].*?[\)\]]", "", s)
    # Entferne Sonderzeichen außer Leerzeichen
    s = re.sub(r"[^\w\s]", "", s)
    # Entferne "the" am Anfang
    s = re.sub(r"^the\s+", "", s)
    # Entferne Suffixe wie "remastered", "live", "remix"
    s = re.sub(r"\s+(live|remix|remaster(ed)?|edit|version|radio edit).*$", "", s)
    # Normalisiere Whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compute_match_confidence(
    deezer_artist: str,
    deezer_title: str,
    lastfm_artist: str,
    lastfm_title: str,
) -> float:
    """
    Berechnet Match-Konfidenz zwischen Deezer und Last.fm Track.
    Returns: 0.0 - 1.0
    """
    artist_score = fuzz.ratio(
        normalize_string(deezer_artist),
        normalize_string(lastfm_artist),
    )
    title_score = fuzz.ratio(
        normalize_string(deezer_title),
        normalize_string(lastfm_title),
    )
    
    # Title ist wichtiger als Artist (60/40)
    combined = 0.4 * artist_score + 0.6 * title_score
    return combined / 100.0


@dataclass
class LastFMResult:
    """Ergebnis einer Last.fm-Abfrage."""
    playcount: Optional[int] = None
    listeners: Optional[int] = None
    matched: bool = False
    match_confidence: float = 0.0
    lastfm_artist: str = ""
    lastfm_title: str = ""
    error: Optional[str] = None


def fetch_lastfm_data(
    network: pylast.LastFMNetwork,
    artist: str,
    title: str,
) -> LastFMResult:
    """
    Holt Track-Daten von Last.fm.
    
    Args:
        network: pylast LastFMNetwork-Instanz
        artist: Artist-Name (von Deezer)
        title: Track-Titel (von Deezer)
    
    Returns:
        LastFMResult mit playcount, listeners und Match-Info
    """
    result = LastFMResult()
    
    for attempt in range(MAX_RETRIES):
        try:
            track = network.get_track(artist, title)
            
            # Track-Info abrufen
            playcount = track.get_playcount()
            listeners = track.get_listener_count()
            
            # Korrigierte Namen von Last.fm (für Match-Confidence)
            corrected_artist = track.get_artist().get_name()
            corrected_title = track.get_title()
            
            # Match-Confidence berechnen
            confidence = compute_match_confidence(
                artist, title,
                corrected_artist, corrected_title,
            )
            
            result.playcount = playcount
            result.listeners = listeners
            result.matched = confidence >= (MATCH_THRESHOLD / 100.0)
            result.match_confidence = confidence
            result.lastfm_artist = corrected_artist
            result.lastfm_title = corrected_title
            
            return result
            
        except pylast.WSError as e:
            if "Track not found" in str(e):
                result.error = "not_found"
                return result
            elif "Rate limit" in str(e).lower():
                time.sleep(5)
                continue
            else:
                result.error = str(e)
                return result
                
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
                continue
            result.error = str(e)
            return result
    
    return result


# ══════════════════════════════════════════════════════════════════════════════
# HAUPTLOGIK
# ══════════════════════════════════════════════════════════════════════════════

def enrich_tracks(
    input_path: Path,
    output_path: Path,
    api_key: str,
    api_secret: str = "",
    resume: bool = True,
) -> dict:
    """
    Reichert Track-Daten mit Last.fm-Metriken an.
    
    Args:
        input_path: Pfad zu scouted_tracks.csv
        output_path: Pfad für Output-CSV
        api_key: Last.fm API Key
        api_secret: Last.fm API Secret (optional)
        resume: Wenn True, überspringe bereits verarbeitete Tracks
    
    Returns:
        Dict mit Statistiken
    """
    # Last.fm-Netzwerk initialisieren
    network = pylast.LastFMNetwork(
        api_key=api_key,
        api_secret=api_secret if api_secret else None,
    )
    
    # Input laden
    df = pd.read_csv(input_path)
    print(f"Geladen: {len(df)} Tracks aus {input_path}")
    
    # Neue Spalten initialisieren (falls nicht vorhanden)
    new_columns = {
        "lastfm_playcount": None,
        "lastfm_listeners": None,
        "lastfm_matched": False,
        "lastfm_match_confidence": 0.0,
        "lastfm_artist": "",
        "lastfm_title": "",
        "lastfm_error": "",
    }
    
    for col, default in new_columns.items():
        if col not in df.columns:
            df[col] = default
    
    # Resume-Logik: Finde unverarbeitete Tracks
    if resume and output_path.exists():
        existing_df = pd.read_csv(output_path)
        # Merge um bereits verarbeitete zu übernehmen
        if "lastfm_playcount" in existing_df.columns:
            processed_ids = set(existing_df[existing_df["lastfm_matched"].notna()]["track_id"])
            print(f"Resume: {len(processed_ids)} bereits verarbeitet")
        else:
            processed_ids = set()
    else:
        processed_ids = set()
    
    # Statistiken
    stats = {
        "total": len(df),
        "processed": 0,
        "matched": 0,
        "not_found": 0,
        "low_confidence": 0,
        "errors": 0,
        "skipped": len(processed_ids),
    }
    
    # Verarbeitung
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Last.fm Enrichment"):
        track_id = row["track_id"]
        
        # Skip wenn bereits verarbeitet
        if track_id in processed_ids:
            continue
        
        artist = row.get("artist", "")
        title = row.get("track_name", "")
        
        if not artist or not title:
            stats["errors"] += 1
            continue
        
        # Last.fm abfragen
        result = fetch_lastfm_data(network, artist, title)
        time.sleep(REQUEST_DELAY)
        
        # Ergebnis speichern
        df.at[idx, "lastfm_playcount"] = result.playcount
        df.at[idx, "lastfm_listeners"] = result.listeners
        df.at[idx, "lastfm_matched"] = result.matched
        df.at[idx, "lastfm_match_confidence"] = result.match_confidence
        df.at[idx, "lastfm_artist"] = result.lastfm_artist
        df.at[idx, "lastfm_title"] = result.lastfm_title
        df.at[idx, "lastfm_error"] = result.error or ""
        
        # Statistiken
        stats["processed"] += 1
        if result.error == "not_found":
            stats["not_found"] += 1
        elif result.error:
            stats["errors"] += 1
        elif result.matched:
            stats["matched"] += 1
        else:
            stats["low_confidence"] += 1
        
        # Zwischenspeichern alle 100 Tracks
        if stats["processed"] % 100 == 0:
            df.to_csv(output_path, index=False)
            tqdm.write(f"  Zwischenstand gespeichert ({stats['processed']} verarbeitet)")
    
    # Final speichern
    df.to_csv(output_path, index=False)
    
    return stats


def print_stats(stats: dict):
    """Gibt Statistiken formatiert aus."""
    print(f"""
╔═══════════════════════════════════════════════════════════════════════════════
║  LAST.FM ENRICHMENT COMPLETE
╠═══════════════════════════════════════════════════════════════════════════════
║  Tracks gesamt:        {stats['total']:>6}
║  Bereits verarbeitet:  {stats['skipped']:>6}
║  Neu verarbeitet:      {stats['processed']:>6}
╠═══════════════════════════════════════════════════════════════════════════════
║  ✅ Gematcht:          {stats['matched']:>6}  ({100*stats['matched']/max(1,stats['processed']):.1f}%)
║  ❓ Nicht gefunden:    {stats['not_found']:>6}  ({100*stats['not_found']/max(1,stats['processed']):.1f}%)
║  ⚠️  Low Confidence:    {stats['low_confidence']:>6}  ({100*stats['low_confidence']/max(1,stats['processed']):.1f}%)
║  ❌ Fehler:            {stats['errors']:>6}
╚═══════════════════════════════════════════════════════════════════════════════
""")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Reichert Deezer-Tracks mit Last.fm-Metriken an"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input CSV (scouted_tracks.csv)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/scouted_tracks_enriched.csv"),
        help="Output CSV (default: data/scouted_tracks_enriched.csv)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Nicht fortsetzen, alle Tracks neu verarbeiten",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Pfad zur .env-Datei mit API-Keys",
    )
    args = parser.parse_args()
    
    # .env laden
    load_dotenv(args.env_file)
    
    api_key = os.getenv("LASTFM_API_KEY")
    api_secret = os.getenv("LASTFM_API_SECRET", "")
    
    if not api_key:
        print("Fehler: LASTFM_API_KEY nicht in .env gefunden!")
        print("Erstelle eine .env-Datei mit:")
        print("  LASTFM_API_KEY=dein_api_key")
        print("")
        print("API-Key erstellen: https://www.last.fm/api/account/create")
        sys.exit(1)
    
    # Input prüfen
    if not args.input.exists():
        print(f"Fehler: Input-Datei nicht gefunden: {args.input}")
        sys.exit(1)
    
    # Output-Verzeichnis erstellen
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Enrichment durchführen
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Resume: {'Nein' if args.no_resume else 'Ja'}")
    print("")
    
    stats = enrich_tracks(
        input_path=args.input,
        output_path=args.output,
        api_key=api_key,
        api_secret=api_secret,
        resume=not args.no_resume,
    )
    
    print_stats(stats)


if __name__ == "__main__":
    main()
