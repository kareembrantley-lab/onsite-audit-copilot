"""
Unit tests for the keyword-matching engine, checklist tagging, severity
escalation, and findings aggregation. Several of these are regression tests
for real bugs caught while building the synthetic demo (see comments below)
so they can't come back.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from onsite_audit_copilot.checklist import _rules_recommend, _rules_tag, list_checklists, load_checklist
from onsite_audit_copilot.entity_extraction import extract_entities
from onsite_audit_copilot.findings import build_findings
from onsite_audit_copilot.keyword_match import best_match, find_matches
from onsite_audit_copilot.schema import ChecklistItem, ExtractedEntities, PhotoMatch, TaggedNote, escalate
from onsite_audit_copilot.severity import compute_severity


def test_find_matches_negation_guard_drops_negated_mentions():
    """Regression test: a naive `keyword in text` check can't tell "a
    backlog piling up" from "no backlog there, running fine" -- both
    contain the literal substring "backlog". Before the negation guard was
    added, a note explicitly saying a bottleneck condition was ABSENT was
    tagged as though it were present, inflating the finding's occurrence
    count with a false positive."""
    assert find_matches("there's a serious backlog here", ["backlog"]) == ["backlog"]
    assert find_matches("no backlog there, running fine", ["backlog"]) == []
    assert find_matches("we haven't seen a trip hazard", ["trip hazard"]) == []


def test_best_match_prefers_more_specific_phrase_on_hit_count_tie():
    """Regression test: "waiting on a part" (toc_exploit_constraint) and
    "waiting on" (toc_identify_constraint) both register exactly one hit
    against the note "...waiting on a part...". The first version of the
    tagger picked whichever checklist item was listed first in the YAML on
    a count tie, which meant an exploit-the-constraint note was silently
    misfiled as identify-the-constraint. Scoring by (hit count, longest
    matched phrase) instead makes the more specific match win."""
    candidates = [
        ("toc_identify_constraint", ["waiting on", "queue"]),
        ("toc_exploit_constraint", ["waiting on a part", "changeover"]),
    ]
    key, hits = best_match("shrink-wrap stoppage, waiting on a part for the machine", candidates)
    assert key == "toc_exploit_constraint"
    assert hits == ["waiting on a part"]


def test_best_match_returns_none_when_nothing_matches():
    """Regression test: an early version of the (hit_count, specificity)
    scoring used (-1, -1) as the initial floor, which a genuine zero-hit
    candidate's (0, 0) score beat -- so the FIRST checklist item always
    "matched" even when nothing in its keyword list appeared in the text."""
    candidates = [("a", ["xyz"]), ("b", ["abc"])]
    key, hits = best_match("this text contains neither keyword", candidates)
    assert key is None
    assert hits == []


def test_escalate_is_a_noop_at_the_top_of_the_scale():
    assert escalate("critical", 1) == "critical"
    assert escalate("high", 1) == "critical"
    assert escalate("low", 2) == "high"  # low -> medium -> high, two steps toward the top of the scale


def test_severity_escalations_stack():
    """A recurring (>=3 occurrences) finding that also independently reads
    as a safety issue escalates twice: once for being safety-flagged, once
    for recurring. This is what lets a "medium" housekeeping item (a
    repeated trip hazard) reach "critical" without needing its own default
    severity inflated -- see the Receiving Dock demo visit."""
    item = ChecklistItem(id="x", category="Housekeeping", prompt="...", default_severity="medium")
    severity, reasons = compute_severity(item, occurrence_count=3, any_note_flagged_safety=True)
    assert severity == "critical"
    assert len(reasons) == 2


def test_severity_escalation_requires_the_recurrence_threshold():
    item = ChecklistItem(id="x", category="Housekeeping", prompt="...", default_severity="medium")
    severity, reasons = compute_severity(item, occurrence_count=2, any_note_flagged_safety=False)
    assert severity == "medium"
    assert reasons == []


def test_checklist_recommend_defaults_to_safety_with_no_keyword_signal():
    """When an engagement description gives no keyword signal at all, the
    rules fallback defaults to the safety & compliance checklist rather
    than guessing between the organization and throughput lenses -- a
    missed safety issue is the costliest kind of wrong default."""
    configs = list_checklists()
    config, rationale, method = _rules_recommend("Just walking around today.", configs)
    assert config.key == "safety_compliance"
    assert method == "rules"


def test_checklist_recommend_matches_engagement_keywords():
    configs = list_checklists()
    config, _, _ = _rules_recommend("Need to find the bottleneck before peak season.", configs)
    assert config.key == "toc_constraint_walk"


def test_rules_tag_does_not_force_a_match():
    checklist = load_checklist("lean_5s")
    entities = extract_entities("Everything here looks completely normal, nothing to report.")
    tagged = _rules_tag("Everything here looks completely normal, nothing to report.", entities, checklist, "n1")
    assert tagged.checklist_item_id is None
    assert tagged.method == "unmatched"


def test_build_findings_preserves_note_id_for_traceability():
    """Regression test: an early version of Finding.evidence stored plain
    text excerpts with no note_id, so the report's appendix (whose entire
    point is tracing a finding back to its source note) rendered "--" in
    the Note ID column for every row. EvidenceItem carries note_id through
    from the TaggedNote/PhotoMatch that produced it."""
    checklist = load_checklist("safety_compliance")
    tagged = TaggedNote(
        note_id="C99", source_text="No guard on the conveyor pinch point.",
        checklist_item_id="safety_machine_guarding", confidence="high",
        rationale="test", entities=ExtractedEntities(), method="rules",
    )
    findings, unmatched = build_findings([tagged], [], checklist)
    assert unmatched == 0
    assert len(findings) == 1
    assert findings[0].evidence[0].note_id == "C99"


def test_build_findings_counts_unmatched_notes_and_photos():
    checklist = load_checklist("safety_compliance")
    tagged = TaggedNote(
        note_id="C01", source_text="text", checklist_item_id=None, confidence="low",
        rationale="no match", entities=ExtractedEntities(), method="unmatched",
    )
    photo = PhotoMatch(note_id="C02", image_path="x.png", caption="caption", checklist_item_id=None, confidence="low", method="fixture_metadata")
    findings, unmatched = build_findings([tagged], [photo], checklist)
    assert findings == []
    assert unmatched == 2
