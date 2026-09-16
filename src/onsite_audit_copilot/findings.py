"""
Aggregates tagged notes and photo matches into Finding objects (one per
checklist item that was actually hit), builds the prioritized action list,
and writes the executive summary. This is the one module where text
evidence and photo evidence come back together after vision.py and
checklist.py processed them on separate tracks.
"""

from __future__ import annotations

from pathlib import Path

from .entity_extraction import SAFETY_KEYWORDS
from .keyword_match import find_matches
from .llm import llm_available, synthesize_executive_summary_llm
from .schema import (
    SEVERITY_ORDER, ActionItem, ChecklistConfig, EvidenceItem, Finding, PhotoMatch, TaggedNote,
)
from .severity import compute_severity

_SEVERITY_ACTION_FRAMING = {
    "critical": "Stop-the-line priority -- address before the next shift starts.",
    "high": "Resolve within the week; assign an owner today.",
    "medium": "Schedule a fix within the next 30 days.",
    "low": "Fold into the next routine standard-work update.",
    "observation": "No action required; noted for trend tracking only.",
}


def _caption_reads_safety(caption: str) -> bool:
    return bool(find_matches(caption.lower(), SAFETY_KEYWORDS))


def build_findings(tagged_notes: list[TaggedNote], photo_matches: list[PhotoMatch], checklist: ChecklistConfig) -> tuple[list[Finding], int]:
    groups: dict[str, dict[str, list]] = {}
    unmatched = 0

    for tn in tagged_notes:
        if tn.checklist_item_id is None:
            unmatched += 1
            continue
        groups.setdefault(tn.checklist_item_id, {"notes": [], "photos": []})["notes"].append(tn)

    for pm in photo_matches:
        if pm.checklist_item_id is None:
            unmatched += 1
            continue
        groups.setdefault(pm.checklist_item_id, {"notes": [], "photos": []})["photos"].append(pm)

    findings: list[Finding] = []
    for item_id, group in groups.items():
        item = checklist.get_item(item_id)
        if item is None:
            continue  # defensive: a stale item id should never reach here, but never crash the report over it
        notes = group["notes"]
        photos = group["photos"]
        occurrence_count = len(notes) + len(photos)

        any_safety = any(n.entities.safety_issue for n in notes) or any(_caption_reads_safety(p.caption) for p in photos)
        severity, reasons = compute_severity(item, occurrence_count, any_safety)

        evidence = [EvidenceItem(note_id=n.note_id, kind="text", detail=n.source_text) for n in notes]
        evidence += [
            EvidenceItem(note_id=p.note_id, kind="photo", detail=p.caption, image_filename=Path(p.image_path).name)
            for p in photos
        ]
        narrative = _build_narrative(item, occurrence_count, evidence)

        findings.append(Finding(
            checklist_item_id=item_id, category=item.category, title=item.category,
            severity=severity, occurrence_count=occurrence_count, evidence=evidence,
            narrative=narrative, escalation_reasons=reasons,
        ))

    findings.sort(key=lambda f: (SEVERITY_ORDER.index(f.severity), -f.occurrence_count))
    return findings, unmatched


def _build_narrative(item, occurrence_count: int, evidence: list[EvidenceItem]) -> str:
    parts = [f"{occurrence_count} instance(s) matched '{item.category}' ({item.prompt.lower()})."]
    text_evidence = [e.detail for e in evidence if e.kind == "text"]
    photo_evidence = [e.image_filename for e in evidence if e.kind == "photo"]
    if text_evidence:
        sample = "; ".join(e.strip().rstrip(".") for e in text_evidence[:2])
        parts.append(f"Field notes: {sample}.")
    if photo_evidence:
        parts.append(f"Photo evidence: {', '.join(photo_evidence[:3])}.")
    return " ".join(parts)


def build_action_list(findings: list[Finding]) -> list[ActionItem]:
    return [
        ActionItem(
            priority_rank=rank, finding_title=f.title, severity=f.severity,
            recommended_action=_SEVERITY_ACTION_FRAMING.get(f.severity, "Review and prioritize."),
        )
        for rank, f in enumerate(findings, start=1)
    ]


def synthesize_executive_summary(facility_name: str, visit_area: str, checklist: ChecklistConfig, findings: list[Finding]) -> tuple[str, str]:
    severity_counts = {level: sum(1 for f in findings if f.severity == level) for level in SEVERITY_ORDER}
    top_titles = [f.title for f in findings[:2]]

    if llm_available():
        summary = synthesize_executive_summary_llm(
            facility_name, visit_area, checklist.name,
            {"severity_counts": severity_counts, "top_findings": top_titles, "total_findings": len(findings)},
        )
        if summary:
            return summary, "llm"

    return _template_summary(facility_name, visit_area, checklist, findings, severity_counts), "template"


def _template_summary(facility_name: str, visit_area: str, checklist: ChecklistConfig, findings: list[Finding], severity_counts: dict) -> str:
    if not findings:
        return (
            f"The {checklist.name.lower()} of {facility_name} -- {visit_area} found no findings against this "
            f"checklist. No action items are outstanding from this visit."
        )

    critical_or_high = severity_counts["critical"] + severity_counts["high"]
    lead = findings[0]

    if critical_or_high == 0:
        headline = f"No critical or high-severity findings from this {checklist.name.lower()}."
    else:
        headline = (
            f"{critical_or_high} critical/high-severity finding(s) from this {checklist.name.lower()}, "
            f"led by '{lead.title}' ({lead.occurrence_count} occurrence(s))."
        )

    total = sum(severity_counts.values())
    return (
        f"{headline} {total} finding(s) total across {facility_name} -- {visit_area}. "
        f"Top priority: {_SEVERITY_ACTION_FRAMING.get(lead.severity, 'review and prioritize')} "
        f"See the prioritized action list below for the full sequence."
    )
