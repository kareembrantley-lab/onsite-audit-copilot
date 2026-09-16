# Architecture notes

This is the module-by-module detail behind the diagram in the [README](../README.md). Each section lists the data contract in and out, so you can unit-test or swap any one stage without reading the rest of the pipeline.

## 1. `schema.py`

The data contracts every other module passes between stages: `FieldNoteInput` (raw), `TranscribedNote`, `ExtractedEntities`, `ChecklistConfig`/`ChecklistItem`, `TaggedNote`, `PhotoMatch`, `EvidenceItem`, `Finding`, `ActionItem`, `FindingsReport`. `SEVERITY_ORDER` and `escalate()` live here too, as a plain ordered list rather than an `IntEnum`, so `severity.py` can do simple index arithmetic ("move one step toward critical") without a parallel mapping table to keep in sync. `escalate()` on `"critical"` is a documented no-op, not an error — callers apply it speculatively without checking first.

`Finding.evidence` is a list of `EvidenceItem` (`note_id`, `kind`, `detail`, optional `image_filename`), not plain text excerpts. An earlier version stored evidence as bare strings, which meant the report's traceability appendix — whose entire point is pointing a reader back to the exact source note — rendered `"--"` in the Note ID column for every row. `tests/test_pipeline.py::test_build_findings_preserves_note_id_for_traceability` is the regression test.

## 2. `stt.py`

**In:** a note id, plus either `audio_path` (live) or `fixture_transcript` (demo).
**Out:** a `TranscribedNote` with `method` set to `"live_whisper"` or `"fixture"`.

Live mode calls OpenAI's Whisper transcription API — a licensed, official hosted endpoint, the same category of choice as `places_signal.py`'s use of Google Places in the sibling `smb-public-signal-audit-agent` project. The demo deliberately does not synthesize fake speech audio; generating audio realistic enough to be worth transcribing is its own project, and faking that step would test nothing real. Instead the synthetic dataset ships pre-transcribed dictation text — the honest equivalent of "the auditor's phone already transcribed this in the field" — and every `TranscribedNote` discloses which path produced it via `method`, so a reviewer never mistakes a fixture transcript for a live one.

`_confidence_from_length()` is a conservative proxy in the absence of Whisper's own segment-level confidence: a very short transcript is more likely a dropped recording or a mis-hearing than a fully captured observation.

## 3. `vision.py`

**In:** a note id, an image path, the active `ChecklistConfig`, and optionally a `fixture_caption` dict.
**Out:** a `PhotoMatch` (`caption`, `checklist_item_id`, `confidence`, `method`).

If `ANTHROPIC_API_KEY` is set, this calls Claude's multimodal input directly on the photo — no separate vision-model dependency alongside the text LLM calls in `llm.py`. Without a key, it falls back to the sidecar metadata file shipped next to each synthetic photo (`scene_description` + `expected_checklist_item`), which is the honest stand-in for "the caption the auditor typed when they took the photo," not a fabricated AI output. This module never sends a photo anywhere except the configured LLM provider.

## 4. `entity_extraction.py`

**In:** a transcribed note's text.
**Out:** `ExtractedEntities` (`equipment`, `process_step`, `safety_issue` + matched keywords, `bottleneck_observation` + matched keywords).

LLM-assisted (`llm.extract_entities_llm`) with a keyword-rules fallback (`EQUIPMENT_VOCAB`, `PROCESS_STEP_VOCAB`, `SAFETY_KEYWORDS`, `BOTTLENECK_KEYWORDS`). The rules path matters more here than in a pure sentiment tool: a missed safety or bottleneck flag doesn't just weaken a score, it can mean a real finding never reaches the report at all, so the keyword lists are built recall-first — `checklist.py`'s tagging step, not this one, is where precision matters. Even when the LLM path is used, the module still recomputes the keyword-matched evidence lists locally, since an LLM call returns a boolean, not which words tripped it, and the report cites the specific matched phrase.

## 5. `keyword_match.py`

**In:** lowercased text and a keyword/candidate list.
**Out:** `find_matches()` returns the matched keyword subset; `best_match()` picks the best-scoring candidate across several keyword lists at once.

Shared by both `entity_extraction.py` and `checklist.py`'s rules fallback, and the site of two real bugs caught while building the demo (see the README's "A bug worth naming" and the two regression tests in `tests/test_pipeline.py`):

1. **Negation blindness.** A plain `keyword in text` check can't distinguish "a backlog piling up" from "no backlog there, running fine" — both contain the substring "backlog." `find_matches()` now looks a short window backward from each match for a negation cue (`"no "`, `"wasn't "`, `"haven't "`, etc.) and drops the match if one is found. This is not sound over arbitrary sentence structure — "not unlike a bottleneck" would still slip through — and that limitation is disclosed rather than hidden; it correctly handles the common, cleanly worded case a field auditor's dictation actually produces, which is what matters here.
2. **First-listed-item bias on ties.** The first version of the tagger picked whichever checklist item scored the most raw keyword hits, with ties broken by iteration order over the YAML file — so a note matching both "waiting on" (1 hit, `toc_identify_constraint`) and the more specific "waiting on a part" (1 hit, `toc_exploit_constraint`) always fell to whichever item was declared first in the config, regardless of which phrase was the better match. `best_match()` scores each candidate by `(hit_count, length_of_longest_matched_keyword)`, so the more specific phrase wins a tie. A related bug surfaced while fixing this: the initial scoring floor was `(-1, -1)`, which a genuine zero-hit candidate's `(0, 0)` score beat, meaning the first candidate always "won" even with no real match at all — fixed by skipping any candidate with zero hits outright rather than scoring it.

## 6. `checklist.py`

**In:** a checklist key (explicit) or an engagement description (for `recommend_checklist()`), plus a transcribed note + its `ExtractedEntities`.
**Out:** a loaded `ChecklistConfig`, or a `TaggedNote`.

Loads the three YAML configs in `data/checklists/` (each with `id`/`category`/`prompt`/`default_severity`/`keywords` per item, plus a top-level `recommend_when` keyword list). `recommend_checklist()` is the build spec's stretch goal: LLM-assisted (`llm.recommend_checklist_llm`) with a keyword-rules fallback that, on zero keyword signal, deliberately defaults to the safety & compliance checklist rather than guessing between the organization and throughput lenses — a missed safety issue is the costliest kind of wrong default. `tag_note_to_checklist()` uses the same LLM-then-rules pattern as every other stage, with the rules path going through `keyword_match.best_match()` described above. An unmatched note gets `checklist_item_id=None` and is excluded from findings rather than force-fit — see `tests/test_pipeline.py::test_rules_tag_does_not_force_a_match`.

## 7. `severity.py`

**In:** a `ChecklistItem` (for its `default_severity`), an occurrence count, and whether any supporting note was independently flagged as a safety issue.
**Out:** `(severity, escalation_reasons)`.

Two escalation rules, both disclosed on the `Finding` itself so a reviewer can see exactly why a "medium" became a "critical" rather than taking the number on faith: a safety-flagged supporting note escalates one level, and 3+ independent occurrences of the same finding escalates one level (a single cluttered corner is a "low"; the same category showing up five times across the floor is a pattern). The two rules can stack — a recurring safety issue escalates twice — which is intentional and is exactly what the Receiving Dock demo visit's Housekeeping finding shows (medium → high → critical). Escalating an already-`"critical"` item is a no-op, not an error, which is what lets the Machine Guarding and Emergency Egress items in the same demo visit absorb both rules without going out of range.

## 8. `findings.py`

**In:** the full list of `TaggedNote` and `PhotoMatch` for a visit, plus the active `ChecklistConfig`.
**Out:** a list of `Finding` (one per checklist item that was actually hit, sorted worst-severity-first), a prioritized `ActionItem` list, and the executive summary.

This is where text evidence (from `checklist.py`) and photo evidence (from `vision.py`) come back together after being processed on separate tracks. `_caption_reads_safety()` runs a photo's caption back through the same negation-aware `SAFETY_KEYWORDS` matcher used in `entity_extraction.py`, so a critical photo with no accompanying voice note can still trigger the safety-escalation rule in `severity.py`. `synthesize_executive_summary()` follows the same LLM-then-template pattern as every other narrative step in this pipeline; the template fallback (`_template_summary()`) explicitly says "No critical or high-severity findings" when that's true, rather than always framing the weakest category as a concern regardless of how mild it actually reads.

## 9. `report.py`

**In:** a `FindingsReport`.
**Out:** a one-page-first `.docx`: title, executive summary (with its synthesis method disclosed), findings grouped by severity with color-coded labels and escalation reasons, a prioritized action list table, a methodology/scope disclosure, and a full note/photo-to-finding traceability appendix.

`pipeline.py` optionally shells out to headless LibreOffice (`soffice --headless --convert-to pdf`) to also produce a PDF; if LibreOffice isn't on `PATH`, this step is skipped and the `.docx` alone is still delivered.

## 10. `llm.py`

The only place `ANTHROPIC_API_KEY` is read, for both text and vision calls (Claude is multimodal, so no separate vision-model dependency is needed). Five call sites — entity extraction, checklist tagging, checklist recommendation, photo captioning, executive-summary synthesis — every one wrapped so an API error, a missing key, or malformed JSON returns `None` rather than raising, with every caller falling back to its rules-based or template equivalent.

## 11. `pipeline.py`

The orchestrator. `run_audit(config, out_dir)` resolves the checklist (explicit or recommended), transcribes and tags every voice note, captions and matches every photo, aggregates into findings, builds the action list and executive summary, and renders the report — returning a dict with every intermediate object so the CLI, a test, or a future API wrapper can inspect any of it without re-parsing the `.docx`.
