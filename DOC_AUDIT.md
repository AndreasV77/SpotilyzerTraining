# Documentation Audit Workflow

A repeatable process for keeping the docs of **both** repositories
(Spotilyzer + SpotilyzerTraining) consistent without introducing new errors
during the cleanup itself.

This file is identical in both repos. Keep it in sync; if the two ever diverge,
the copy in **SpotilyzerTraining** is canonical.

---

## Why this exists

Past doc fixes failed in a specific way: a single pass read, reasoned about
solutions, and edited several files at once — and in that mixed state it
straightened one document, lightly creased a second, and planted a fresh error
in a third. The fix process was itself the error source.

The cure is separation. Auditing and fixing are different jobs, run in different
passes, never mixed. Reasoning about *how* to solve a defect does not happen
during an audit (it pulls focus away from finding defects) and barely happens
during a fix (the solution is usually obvious once the defect is isolated).

---

## Authority hierarchy (never reorder)

When two sources disagree, the higher one wins. Lower sources are corrected to
match — never the reverse.

1. `outputs/reports/evaluation_report_*.json`, `training_report_*.json`
   — ground truth for every metric. Numbers are read from here, never estimated.
2. `SpotilyzerTraining/CLAUDE.md` — authoritative for model / training / dataset facts.
3. `Spotilyzer/CLAUDE.md` — authoritative for app / pipeline / GUI / CLI facts.
4. `README.md` (both repos) — derived. Lowest priority. Audited last. For a
   private repo, may be left OPEN until a public release is planned.

**Cross-repo flow:**
- Training facts flow Training → Main, never the reverse.
- App / pipeline facts flow Main → Training, never the reverse.
- Any number that appears in both repos must cite the **same** source report.
  (This is what structurally prevents the "86.9 here, 82.5 there" class of bug.)

---

## Audit pass — READ ONLY

No edits. No solution proposals. No "while I'm here" fixes.

For each document, in hierarchy order (1 → 4):
1. Compare every metric against its source report (level 1).
2. Compare every shared fact against the authoritative repo for that fact.
3. For each mismatch, append a row to `ERRATA.md`:
   `ID | found-date | file | defect | OPEN`
   State the defect only — what is wrong, not how to fix it.

### Repo hygiene (also part of this pass)

Not a documentation defect in the narrow sense, but the same read-only rule applies:
find it, log it to `ERRATA.md`, do not touch it. Reason this belongs here: an
accidental local deletion followed by "recovery" from the remote/web repo silently
resurrects whatever bad state the remote happens to have — a detached HEAD, a
tracked file that should never have been tracked, or unpushed local work that the
remote never saw in the first place. Cheap to catch, expensive to discover by
accident later. Check, per repo:

- **Branch attachment** — is HEAD attached to a branch, not detached
  (`git branch --show-current` / `git status`)? A detached HEAD means the next
  commit risks becoming unreachable the moment someone checks out a branch.
- **Sync state** — local vs. `origin`: ahead, behind, or diverged
  (`git status -sb`, `git rev-list --count origin/main..main` and the reverse)?
  Unpushed commits are invisible to anything reading only the remote (a GitHub
  connector, a fresh clone) and are exactly what an accidental-recovery scenario
  would lose.
- **Tracked-file / .gitignore gaps** — any file under version control that an
  ignore pattern was clearly meant to exclude but doesn't (e.g. a glob that
  doesn't match a subdirectory)? Check with `git ls-files` against the intent of
  `.gitignore`, not just its literal patterns. Large or generated binaries
  slipping into tracked history are the concrete risk.
- **Working tree cleanliness** — `git status`: anything uncommitted sitting
  around across sessions that shouldn't be?

Log each finding the same way as a doc defect (same `ERRATA.md`, same ID
prefix) — the `file` column can name a path, a glob pattern, or just "git state"
if there's no single file to point at.

The pass ends when all documents are checked. Nothing has been edited.

If the audit was prompted as part of a larger request, it still ends here.
"Audit and fix it" is two passes, not one — run the audit, then start a fresh
fix pass.

---

## Fix pass — SEPARATE invocation

Ideally a separate session, so the fixer holds `ERRATA.md` as a closed list and
is not also hunting for new defects.

1. Re-verify the top OPEN entry against the current file (it may already be
   resolved, or the file may have moved on since the audit).
2. Resolve exactly that one defect — one file edit for a doc defect, or the
   equivalent single git operation for a repo-hygiene finding (reattach a
   branch, push, `git rm --cached` + fix the ignore pattern). Confirm the
   before-state first (e.g. check the remote hasn't moved since the audit)
   rather than assuming it's still accurate.
3. Add one line to `CHANGELOG.md`.
4. Mark the entry FIXED (date) in `ERRATA.md`. Do not delete the row.
5. Next entry.

Do NOT search for new defects during a fix pass. Anything newly spotted is noted
as a one-line OPEN entry in `ERRATA.md` and left for the next audit — it is not
acted on now.

---

## What a good prompt looks like

Audit:   "Run a read-only documentation audit per DOC_AUDIT.md. Log defects to
          ERRATA.md. Do not fix anything, do not propose solutions."
         (This already includes the repo-hygiene checks — they're part of the
         audit pass, not a separate ask.)
Fix:     "Run a fix pass per DOC_AUDIT.md: work ERRATA.md top to bottom, one edit
          per defect, one CHANGELOG line each, mark FIXED. Do not hunt for new
          defects."

Anything that says "review AND fix at once", "update to current web trends", or
"compare against upstream/canary and apply" defeats the entire workflow.
Local reports are the source of truth, not the web. Trend is not a correctness
criterion.

---

## Time stamps

`CHANGELOG.md` and `ERRATA.md` use plain ISO dates (`YYYY-MM-DD`). If a time is
needed, ISO with offset (`2026-05-29T14:30+02:00`). Earth time. Pluto-relative
stardates and other flights of fancy belong in `poetry_collection.md`.
