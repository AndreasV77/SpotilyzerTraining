# Errata — open documentation defects (SpotilyzerTraining)

Defects found during documentation audits. **Record only.**

Rules:
- An audit pass writes defects here. It does NOT fix them and does NOT propose solutions.
- A separate fix pass works this list top to bottom, one defect at a time.
- A defect's `Status` goes OPEN → FIXED (date + CHANGELOG line). Never delete rows; mark them FIXED so the history stays visible.
- IDs are permanent. Prefix `ST-` = this repo. (`SP-` lives in Spotilyzer/ERRATA.md.)

> Audit of 2026-05-29 **verified against the source eval reports** (Level 1):
> `_20260529` = BA 63.0 / Hit 86.9 / Flop 67.5 / Mid 34.7;
> `_20260319` = BA 64.2 / Hit 82.5 / Flop 73.5 / Mid 36.6; both on the 4545-sample holdout.
> CLAUDE.md model figures match the reports. README and one table note do not.

| ID | Found | File | Defect | Status |
|----|-------|------|--------|--------|
| ST-001 | 2026-05-29 | README.md | "Current Model Status" lists only `_20260319` (BA 64.2 / Hit 82.5 / Flop 73.5 — correct) and the `spotify_charts` predecessor. The `_20260529` default is absent. Header "As of 2026-03-19" is stale (CLAUDE.md is at 2026-05-29). | FIXED 2026-05-30 |
| ST-002 | 2026-05-29 | README.md | "Goals" line reads "BA ≥ 65% (64.2%, in progress)" — that is the Alternative's BA. The default `_20260529` is at 63.0%. Figures track the wrong model once ST-001 is addressed. | FIXED 2026-05-30 |
| ST-003 | 2026-05-29 | CLAUDE.md | "Current Model Status" metrics are per-30s holdout segment, not per song. The main project now averages chunk probabilities over the full track. No note that song-level evaluation is still pending. | FIXED 2026-05-30 |
| ST-004 | 2026-05-29 | CLAUDE.md | Model status table has two rows with the identical filename stem `..._20260319` — the depth=4 Alternative (22,722 val.) and the Session-5 model (~8,960 val.). The Session-5 `.joblib` was overwritten and no longer exists. Same filename, two value sets, no explicit note — invites the SP-002 confusion. | FIXED 2026-05-30 |
| ST-005 | 2026-07-13 | CLAUDE.md | Documented command `python scripts/scout_spotify.py --input-dir G:/Dev/SpotilyzerData/spotify/YYYY-MM-DD --dry-run` uses a flag that doesn't exist. The script's actual `--help` only accepts `--input` (confirmed by running it — `--input-dir` errors as unrecognized). | OPEN |
