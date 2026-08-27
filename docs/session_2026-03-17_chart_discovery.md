# Session Summary: Deezer Chart Discovery & Expansion

**Date:** 2026-03-17
**Context:** SpotilyzerTraining — Expanding chart data sources

---

## 1. Starting Point

**Problem:** Hit Recall at ~15% (target: 80%). Root cause: too few Hit samples in training.

**Approach:** Include more chart playlists from various countries in training to grow the Hit class.

---

## 2. Findings

### 2.1 Deezer Chart Accounts

There are several semi-official Deezer accounts that maintain chart playlists automatically:

| Account | User ID | Function |
|---------|---------|----------|
| **Deezer** | 2 | Original Editorial |
| **Deezer Charts** | 637006841 | Automated Top-100 Charts per country |
| **Deezer Editorial** | 2748989402 | Newer editorial account |
| **Deezer Best Of** | 4036701362 | Best-Of Compilations |

**Important:** The Deezer Search API returns **incorrect follower counts** (often 0). Real counts are only available via a direct Playlist API call (`/playlist/{id}`).

### 2.2 Usable Charts (currently verified)

These 22 countries have official, current Deezer charts:

```yaml
# For clusters.yaml — all verified current (March 2026)
charts:
  # Already configured
  DE: {playlist_id: 1111143121}
  US: {playlist_id: 1313621735}
  UK: {playlist_id: 1111142221}
  FR: {playlist_id: 1109890291}
  JP: {playlist_id: 1362508955}
  BR: {playlist_id: 1111141961}
  ES: {playlist_id: 1116190041}
  GLOBAL: {playlist_id: 3155776842}
  
  # NEW — weighting/tier still open
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
  FI: {playlist_id: 1221034071}   # 56K followers (local artists!)
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

### 2.3 Problematic Charts (check manually)

| Country | Problem | Sample Tracks |
|---------|---------|---------------|
| **KR** 🇰🇷 | Classical music at #2/#3 — bot manipulation? | Borodine, Saint-Saëns instead of K-Pop |
| **AR** 🇦🇷 | BTS/Jimin only — K-Pop stan takeover | "Who", "Set Me Free", "Let Me Know" |
| **CL** 🇨🇱 | Old BTS tracks only (2014) — definitely manipulated | "Danger", "24/7=Heaven" |
| **PT** 🇵🇹 | White-noise tracks | "Barulho Para Relaxar" at #1 |
| **TH** 🇹🇭 | Inconsistent mix | French Star Academy at #3 |

### 2.4 Not Usable

| Country | Reason |
|---------|--------|
| **TR** | "Top Turkey 2020" — 6 years outdated |
| **UAE** | User-curated, 2019, 7 followers |
| **NZ** | User-curated, 293 tracks, not a real chart |
| **IN** | No official India chart found (search returns Indonesia) |

---

## 3. Tool Improvements (done)

`scripts/analyze_clusters.py` was extended:

1. **Auto-save:** JSON report is automatically saved to `outputs/reports/`
2. **Real follower counts:** Direct API call instead of Search API
3. **Sample tracks:** First 3 tracks for freshness verification
4. **More Deezer IDs:** 637006841 and 4036701362 recognized as official accounts

---

## 4. Open Tasks

### 4.1 Immediate (next chat)

- [ ] Enter usable charts (22 countries) into `clusters.yaml`
- [ ] Manually review problematic charts (KR, AR, CL, PT, TH) and decide
- [ ] Define weighting system for charts (criteria, not implementation)

### 4.2 Follow-up

- [ ] Re-run `compute_labels.py` after chart expansion
- [ ] Restart training
- [ ] Evaluate Hit Recall

### 4.3 Backlog

- [ ] Manually search for India chart
- [ ] Turkey: manually search for current chart
- [ ] Regional scores & confidence architecture
- [ ] Genre-specific thresholds in `thresholds.yaml`

---

## 5. Work Brief: New Chat

### Goal
Finalize chart expansion and define weighting system for charts.

### Context Files (read in this order)
1. `CLAUDE.md` — Project context
2. `docs\session_2026-03-17_chart_discovery.md` — This summary
3. `outputs\reports\cluster_analysis_20260317_200608.json` — Discovered charts with sample tracks

### Tasks

1. **Define weighting system:**
   - Which criteria determine a chart's weight?
   - Candidates: follower count, market size (IFPI data), genre diversity, manipulation risk
   - Develop concept, no implementation

2. **Evaluate problematic charts:**
   - Go through KR, AR, CL, PT, TH individually
   - Per country: include / exclude / include with warning

3. **Update clusters.yaml:**
   - Enter all usable charts
   - Document weighting info as comments

### Scope Limits (not in this chat)
- No training runs
- No changes to `compute_labels.py` or `train_model.py`
- No GUI discussion
- No regional scores architecture (separate topic)

---

## 6. References

- **JSON Report:** `outputs/reports/cluster_analysis_20260317_200608.json`
- **Tool:** `scripts/analyze_clusters.py --discover-charts`
- **Deezer API:** `https://api.deezer.com/playlist/{id}`

---

**Created:** 2026-03-17T20:15
