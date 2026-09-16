"""
Turns a checklist item's default severity into the finding's actual
severity, given what was actually observed. Two escalation rules, both
disclosed on the Finding itself (`escalation_reasons`) so a reviewer can
see exactly why a "medium" became a "high" rather than taking the number
on faith:

1. Safety escalation: if any note tagged to this item was independently
   flagged as a safety issue by entity_extraction.py, escalate one level.
   This only ever tightens the read -- a checklist item whose default is
   already "critical" (see safety_compliance.yaml's egress/machine-guarding
   items) simply has nowhere higher to go.
2. Recurrence escalation: if the same checklist item was hit by three or
   more independent notes in one visit, escalate one level. A single
   cluttered corner is a "low"; the same category showing up five times
   across the floor is a pattern, not a one-off, regardless of how mild
   any single instance reads.

The two can stack (a recurring safety issue escalates twice), which is
intentional -- that combination is exactly the profile a report should
surface first.
"""

from __future__ import annotations

from .schema import ChecklistItem, escalate

RECURRENCE_THRESHOLD = 3


def compute_severity(item: ChecklistItem, occurrence_count: int, any_note_flagged_safety: bool) -> tuple[str, list[str]]:
    severity = item.default_severity
    reasons: list[str] = []

    if any_note_flagged_safety:
        escalated = escalate(severity, 1)
        if escalated != severity:
            reasons.append("Escalated: at least one supporting note independently read as a safety issue.")
        severity = escalated

    if occurrence_count >= RECURRENCE_THRESHOLD:
        escalated = escalate(severity, 1)
        if escalated != severity:
            reasons.append(f"Escalated: this finding recurred across {occurrence_count} separate notes (threshold: {RECURRENCE_THRESHOLD}).")
        severity = escalated

    return severity, reasons
