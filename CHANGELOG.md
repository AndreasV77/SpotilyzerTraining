# Changelog

Format: `YYYY-MM-DD — <file(s)> — <what changed>` · newest on top.
Code changes reference the short commit hash where available.
Append-only: never rewrite or delete past entries. One change = one line.
History before the first entry below lives in `git log`.

---

## 2026-07-13
- Deploy: `depth4refresh` (depth=4/col=0.6, 24,170 samples) copied to `Spotilyzer/models/` as the new default (SLYZR 1.3), `active_model.txt` updated. `_20260529` (SLYZR 1.2, depth=5/col=0.8) retired to `P:\BACKUP\Archive\Spotilyzer_model_20260529_retired_2026-07-13.zip`. Verified end-to-end via `spotilyzer.cli.analyze` before committing.
- outputs/models/, outputs/reports/ — final triage of the Session 11 sweep: kept `depth4refresh` (standing default, depth=4/col=0.6) and `d3c08` (depth=3/col=0.8, only config to cross BA>=65%; kept for manual spot-testing on varied tracks, not yet a deploy candidate). Archived the remaining four (unlabeled depth=5/col=0.8 refresh, d5c07, d4c07, d4c08) to `P:\BACKUP\Archive\Spotilyzer_model_archive_2026-07-13_batch2.zip` — none offered a distinct trade-off profile beyond what's already documented in the Session 11 comparison table, and d4c08's marginal edge over depth4refresh (BA +0.4pp) was within the ~1pp 5-fold CV noise band.
- scripts/utils/musicbrainz.py — new: mb_get() + get_isrc_by_artist_title() extracted from scout_kworb.py (was duplicated inline), now takes logger as a parameter instead of relying on a module-global. Shared by scout_kworb.py and the new enrich_isrc.py.
- scripts/scout_kworb.py — refactored to import MB helpers from utils/musicbrainz.py instead of defining them locally; no behavior change.
- scripts/enrich_isrc.py — new: resumable ISRC backfill script for tracks missing `isrc` (main dataset never had the field; spotify_charts/kworb were scouted with --skip-mb). Checkpoints isrc_cache.json after every new (non-cached) lookup and tracks.jsonl every --checkpoint-every tracks (default 50); safe to Ctrl+C or hard-kill anytime — re-running re-scans for missing ISRCs and the cache prevents any repeat MusicBrainz calls. Verified via live test run (found real ISRCs, correct resume behavior, survived a hard kill mid-run without corrupting isrc_cache.json or tracks.jsonl). ~28,997 tracks currently missing ISRC across all three datasets (main 9,661 / spotify_charts 960 / kworb 18,376 before this session's refresh scan finishes adding more).
- Kworb rescan + Spotify Charts snapshot (2026-07-13) started — periodic ~2-3 month data refresh, current model ceiling (BA ~63-65%) accepted, no further hyperparameter tuning planned.
- Kworb rescan complete: 18,376 → 19,626 tracks (+1,250). Spotify Charts: 960 → 1,712 tracks (+752, 10 markets incl. fresh au/se snapshots). 1,437 new preview downloads total. MERT-330M embeddings extracted for all new tracks (+1,151 kworb, +276 spotify_charts), 27,431 total.
- configs/training.yaml — corrected: file had max_depth=5/colsample=0.8 persisted despite CLAUDE.md documenting max_depth=4/colsample=0.6 as current — a doc/reality drift that caused an initial mistaken retrain. Standing default is now genuinely max_depth=4/colsample=0.6 in the file (not just in docs).
- Hyperparameter sweep on fresh data (24,170 validated samples, all official evaluate.py numbers): depth=5/col=0.8 (refreshed) BA=62.2/Hit=86.5/Flop=67.7; depth=5/col=0.7 BA=63.1/Hit=86.3/Flop=69.6; depth=4/col=0.6 BA=64.3/Hit=82.4/Flop=74.5; depth=4/col=0.7 BA=64.5/Hit=82.8/Flop=74.7; depth=4/col=0.8 BA=64.7/Hit=82.8/Flop=75.2; depth=3/col=0.8 BA=65.3 (first config to cross the 65% target)/Hit=79.4/Flop=81.0. Decision: depth=4/col=0.6 remains the standing default (depth=5 variants judged a taste call, not a technical win); no config displaced it. Which model (if any) replaces the deployed `_20260529` is not yet decided.
- Repo cleanup: removed `outputs/models/archive/` and `outputs/reports/archive/` (12 old models + ~50 old reports/recon/cluster-analysis files) plus the superseded `_20260319`/`_20260529` model+report files from `outputs/models/` and `outputs/reports/` root. Zipped to `P:\BACKUP\Archive\Spotilyzer_model_archive_2026-07-13.zip` (66 files, 15.5 MB) before removal — nothing lost, just decluttered. `outputs/embeddings/` untouched (reusable training input, not a disposable model output). Mirrored in Spotilyzer/models/ (see that repo's changelog) except the currently-deployed `_20260529`, kept in place to avoid leaving the app without a working model.

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
