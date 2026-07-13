# Changelog

Format: `YYYY-MM-DD — <file(s)> — <what changed>` · newest on top.
Code changes reference the short commit hash where available.
Append-only: never rewrite or delete past entries. One change = one line.
History before the first entry below lives in `git log`.

---

## 2026-07-13
- scripts/utils/musicbrainz.py — new: mb_get() + get_isrc_by_artist_title() extracted from scout_kworb.py (was duplicated inline), now takes logger as a parameter instead of relying on a module-global. Shared by scout_kworb.py and the new enrich_isrc.py.
- scripts/scout_kworb.py — refactored to import MB helpers from utils/musicbrainz.py instead of defining them locally; no behavior change.
- scripts/enrich_isrc.py — new: resumable ISRC backfill script for tracks missing `isrc` (main dataset never had the field; spotify_charts/kworb were scouted with --skip-mb). Checkpoints isrc_cache.json after every new (non-cached) lookup and tracks.jsonl every --checkpoint-every tracks (default 50); safe to Ctrl+C or hard-kill anytime — re-running re-scans for missing ISRCs and the cache prevents any repeat MusicBrainz calls. Verified via live test run (found real ISRCs, correct resume behavior, survived a hard kill mid-run without corrupting isrc_cache.json or tracks.jsonl). ~28,997 tracks currently missing ISRC across all three datasets (main 9,661 / spotify_charts 960 / kworb 18,376 before this session's refresh scan finishes adding more).
- Kworb rescan + Spotify Charts snapshot (2026-07-13) started — periodic ~2-3 month data refresh, current model ceiling (BA ~63-65%) accepted, no further hyperparameter tuning planned.

## 2026-05-30
- CLAUDE.md — added Documentation Workflow pointer (audit/errata/fix cycle)
- CLAUDE.md — ST-004: Session-5 Superseded row annotated: file overwritten by Session-6 retrain with same date stamp (_20260319); no longer exists.
- README.md — ST-001: Added _20260529 as Default model; updated header date to 2026-05-29; _20260319 relabelled Alternative; added Status column.
- README.md — ST-002: Goals BA figure updated from 64.2% (Alternative) to 63.0% (Default _20260529); Hit Recall updated to 86.9%.
- CLAUDE.md — ST-003: Added note that Current Model Status metrics are per 30s clip; song-level evaluation still pending.

## 2026-05-29
- models/ — `_20260529` trained and deployed as default (depth=5, col=0.8; BA 63.0 / Hit 86.9 / Flop 67.5 on 4545-sample holdout). `_20260319` (depth=4) retained as alternative. Superseded models archived.
- CLAUDE.md — Session 10 logged: inference decision (Option D simplified = chunk-averaging); Current Model Status table reworked into default (`_20260529`) + alternative (`_20260319`); date bumped to 2026-05-29.

## (earlier)
- See `git log`. English-translation pass and pre-2026-05-29 session history not retro-logged.
