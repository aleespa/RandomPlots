---
name: next-reel
description: Continue the Reel schedule. Use when the user asks for "the next reel/one", "the next post", "continue the schedule", or a Reel for a scheduled date. Finds the next open slot in SCHEDULE.md, pitches one strong concept within its category, and keeps the row's status current.
---

# Next Reel from the schedule

`SCHEDULE.md` (repo root) is the source of truth: one row per posting date with a **Category**,
a **Script** path, a **Title** and a **Status** (`todo` → `pitched` → `in progress` → `video ok`
→ `music ok` → `posted`). The full Reel workflow is in `docs/AGENT_INSTRUCTIONS.md`; this skill
only adds the schedule on top of it.

## 1. Pick the slot

- If the user names a date, use that row.
- Otherwise, first check for a row that is `pitched`, `in progress`, `video ok` or `music ok`.
  If there is one, say so and ask whether to finish it or start the next one. Don't silently
  skip unfinished work.
- If nothing is open, take the earliest row with status `todo`.

Tell the user the slot in one line: date, day, category and script path.

## 2. Avoid repeats

Read the module docstring titles (the first lines) of the recent scripts in `lab/2026/`
(about the last 30), plus every earlier row of the same category in `SCHEDULE.md`. Don't
pitch an idea that is already there or very close to one.

## 3. Pitch one concept (Phase 1 of the doc)

Pitch **a single concept: the strongest you can find inside the slot's category**. No menu,
no runners-up. Do the comparing yourself, using the category's examples in the Categories
table as a starting point (not a limit), and keep only the candidate that best meets Phase 1's
criteria: a hook in the first second, one clear idea with a surprise, a render that fits the
~1 min budget as planned, and nothing close to what step 2 found. Against the recent Reels,
vary the sub-topic, the aesthetic and palette (dark backgrounds read well in the feed), and
loop vs one-shot.

Use Phase 1's pitch fields, including the one-line **Why this one**. Keep it short enough to
read in a minute. Then set the row's status to `pitched`.

If the user turns it down, take their reason, drop that direction, and pitch the next
strongest single concept.

## 4. Build and verify (Phases 2–3)

Once the user approves the concept, put the title in the row and set the status to `in progress`.
Write the script at the row's path (the posting date, even if today is a different day),
iterate with `RP_FPS=10` previews, then do the full timed render and verification.

- Set `video ok` when the user approves the picture.
- Set `music ok` once a soundtrack is chosen (doc section "Soundtrack").
- `posted` is set by `uv run python -m tools.post_reel <date>` when it publishes, which you
  run only when the user tells you to post (dry-run first). If the user posted by hand,
  set it when they say so.

## 5. Keep the file tidy

Edit only the row in question (and the category of two traded rows if the user swaps them).
Never regenerate the whole file: titles and statuses already filled in must survive.
