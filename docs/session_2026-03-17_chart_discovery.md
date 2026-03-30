# Session Summary: Deezer Chart Discovery & Expansion

**Datum:** 2026-03-17
**Kontext:** SpotilyzerTraining — Erweiterung der Chart-Datenquellen

---

## 1. Ausgangslage

**Problem:** Hit Recall bei ~15% (Ziel: 80%). Hauptursache: zu wenig Hit-Samples im Training.

**Lösungsansatz:** Mehr Chart-Playlists aus verschiedenen Ländern ins Training aufnehmen, um die Hit-Klasse zu vergrößern.

---

## 2. Erkenntnisse

### 2.1 Deezer Chart-Accounts

Es gibt mehrere semi-offizielle Deezer-Accounts, die automatisiert Chart-Playlists pflegen:

| Account | User-ID | Funktion |
|---------|---------|----------|
| **Deezer** | 2 | Original Editorial |
| **Deezer Charts** | 637006841 | Automatische Top-100-Charts pro Land |
| **Deezer Editorial** | 2748989402 | Neuerer Editorial-Account |
| **Deezer Best Of** | 4036701362 | Best-Of-Compilations |

**Wichtig:** Die Deezer Search-API liefert **falsche Follower-Zahlen** (oft 0). Die echten Zahlen bekommt man nur über direkten Playlist-API-Call (`/playlist/{id}`).

### 2.2 Brauchbare Charts (verifiziert aktuell)

Diese 22 Länder haben offizielle, aktuelle Deezer Charts:

```yaml
# Für clusters.yaml — alle verifiziert aktuell (März 2026)
charts:
  # Bereits konfiguriert
  DE: {playlist_id: 1111143121}
  US: {playlist_id: 1313621735}
  UK: {playlist_id: 1111142221}
  FR: {playlist_id: 1109890291}
  JP: {playlist_id: 1362508955}
  BR: {playlist_id: 1111141961}
  ES: {playlist_id: 1116190041}
  GLOBAL: {playlist_id: 3155776842}
  
  # NEU — Gewichtung/Tier noch offen
  IT: {playlist_id: 1116187241}   # 678K followers
  MX: {playlist_id: 1111142361}   # 1.05M followers
  CA: {playlist_id: 1652248171}   # 42K followers
  AU: {playlist_id: 1313616925}   # 59K followers
  PL: {playlist_id: 1266972311}   # 107K followers
  NL: {playlist_id: 1266971851}   # 273K followers
  SE: {playlist_id: 1313620305}   # 69K followers
  AT: {playlist_id: 1313615765}   # 61K followers
  CH: {playlist_id: 1313617925}   # 58K followers
  BE: {playlist_id: 1266968331}   # 152K followers
  NO: {playlist_id: 1313619885}   # 15K followers
  DK: {playlist_id: 1313618905}   # 32K followers
  FI: {playlist_id: 1221034071}   # 56K followers (lokale Künstler!)
  CO: {playlist_id: 1116188451}   # 1.5M followers
  ID: {playlist_id: 1116188761}   # 338K followers
  PH: {playlist_id: 1362518895}   # 57K followers
  ZA: {playlist_id: 1362528775}   # 62K followers
  EG: {playlist_id: 1362501615}   # 111K followers
  SA: {playlist_id: 1362521285}   # 27K followers
  IE: {playlist_id: 1313619455}   # 39K followers
  SG: {playlist_id: 1313620765}   # 21K followers
  MY: {playlist_id: 1362515675}   # 5K followers
```

### 2.3 Problematische Charts (manuell prüfen)

| Land | Problem | Sample-Tracks |
|------|---------|---------------|
| **KR** 🇰🇷 | Klassik auf #2/#3 | Borodine, Saint-Saëns — Bot-Manipulation? |
| **AR** 🇦🇷 | Nur BTS/Jimin | K-Pop-Stan-Takeover |
| **CL** 🇨🇱 | Alte BTS-Tracks (2014) | "Danger", "24/7=Heaven" — definitiv manipuliert |
| **PT** 🇵🇹 | White-Noise-Tracks | "Barulho Para Relaxar" auf #1 |
| **TH** 🇹🇭 | Inkonsistenter Mix | Französische Star Academy auf #3 |

### 2.4 Nicht brauchbar

| Land | Grund |
|------|-------|
| **TR** | "Top Turkey 2020" — 6 Jahre veraltet |
| **UAE** | User-kuratiert, 2019, 7 Follower |
| **NZ** | User-kuratiert, 293 Tracks, keine echte Chart |
| **IN** | Keine offizielle India-Chart gefunden (Suche findet nur Indonesia) |

---

## 3. Tool-Verbesserungen (erledigt)

`scripts/analyze_clusters.py` wurde erweitert:

1. **Auto-Save:** JSON-Report wird automatisch in `outputs/reports/` gespeichert
2. **Echte Follower-Zahlen:** Direkter API-Call statt Search-API
3. **Sample-Tracks:** Erste 3 Tracks zur Aktualitäts-Prüfung
4. **Mehr Deezer-IDs:** 637006841 und 4036701362 als offizielle Accounts erkannt

---

## 4. Offene Aufgaben

### 4.1 Sofort (nächster Chat)

- [ ] Brauchbare Charts (22 Länder) in `clusters.yaml` eintragen
- [ ] Problematische Charts (KR, AR, CL, PT, TH) manuell prüfen und entscheiden
- [ ] Gewichtungssystem für Charts definieren (Kriterien, nicht Implementation)

### 4.2 Folgend

- [ ] `compute_labels.py` neu ausführen nach Chart-Erweiterung
- [ ] Training neu starten
- [ ] Hit Recall evaluieren

### 4.3 Backlog

- [ ] India-Chart manuell suchen
- [ ] Türkei: Aktuelle Chart manuell suchen
- [ ] Regionale Scores & Confidence-Architektur
- [ ] Genre-spezifische Schwellenwerte in `thresholds.yaml`

---

## 5. Arbeitsauftrag: Neuer Chat

### Ziel
Chart-Expansion finalisieren und Gewichtungssystem für Charts definieren.

### Kontext-Dateien (in dieser Reihenfolge lesen)
1. `G:\Dev\Source\SpotilyzerTraining\CLAUDE.md` — Projekt-Kontext
2. `G:\Dev\Source\SpotilyzerTraining\docs\session_2026-03-17_chart_discovery.md` — Diese Zusammenfassung
3. `G:\Dev\Source\SpotilyzerTraining\outputs\reports\cluster_analysis_20260317_200608.json` — Entdeckte Charts mit Sample-Tracks

### Aufgaben

1. **Gewichtungssystem definieren:**
   - Welche Kriterien bestimmen die Gewichtung einer Chart?
   - Kandidaten: Follower-Anzahl, Marktgröße (IFPI-Daten), Genre-Diversität, Manipulations-Risiko
   - Konzept erarbeiten, keine Implementation

2. **Problematische Charts bewerten:**
   - KR, AR, CL, PT, TH einzeln durchgehen
   - Pro Land: Übernehmen / Ausschließen / Mit Warnung übernehmen

3. **clusters.yaml aktualisieren:**
   - Alle brauchbaren Charts eintragen
   - Gewichtungs-Infos als Kommentare dokumentieren

### Scope-Grenzen (nicht in diesem Chat)
- Kein Training starten
- Keine Änderungen an `compute_labels.py` oder `train_model.py`
- Keine GUI-Diskussion
- Keine regionale Scores-Architektur (separates Thema)

---

## 6. Referenzen

- **JSON-Report:** `outputs/reports/cluster_analysis_20260317_200608.json`
- **Tool:** `scripts/analyze_clusters.py --discover-charts`
- **Deezer API:** `https://api.deezer.com/playlist/{id}`

---

**Erstellt:** 2026-03-17T20:15
