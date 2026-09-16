"""
Renders a FindingsReport as a one-page-first Word document: an executive
summary banner, findings grouped by severity (critical first) with their
evidence, a prioritized action list, then a methodology/scope disclosure
and a full note-to-finding traceability appendix -- so a reviewer can trace
any line in the summary back to the exact field note or photo that
produced it.
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

from .schema import FindingsReport

_SEVERITY_COLOR = {
    "critical": RGBColor(0x8B, 0x00, 0x00),
    "high": RGBColor(0xC0, 0x50, 0x0B),
    "medium": RGBColor(0xBF, 0x8F, 0x00),
    "low": RGBColor(0x2E, 0x75, 0xB6),
    "observation": RGBColor(0x60, 0x60, 0x60),
}
_SEVERITY_LABEL = {
    "critical": "CRITICAL", "high": "HIGH", "medium": "MEDIUM",
    "low": "LOW", "observation": "OBSERVATION",
}


def _add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    return h


def build_report(report: FindingsReport, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()

    title = doc.add_heading(f"{report.facility_name} -- {report.visit_area}", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT

    meta = doc.add_paragraph()
    meta.add_run(f"Visit date: {report.visit_date}    Checklist: {report.checklist.name}").italic = True

    if report.synthetic_data_note:
        note = doc.add_paragraph()
        note.add_run(report.synthetic_data_note).italic = True

    _add_heading(doc, "Executive Summary", level=1)
    summary_p = doc.add_paragraph(report.executive_summary)
    method_note = doc.add_paragraph()
    method_note.add_run(
        f"(Synthesis method: {report.executive_summary_method})"
    ).font.size = Pt(8)

    _add_heading(doc, "Findings", level=1)
    if not report.findings:
        doc.add_paragraph("No findings were tagged against this checklist during this visit.")
    for f in report.findings:
        p = doc.add_paragraph()
        run = p.add_run(f"[{_SEVERITY_LABEL[f.severity]}] ")
        run.bold = True
        run.font.color.rgb = _SEVERITY_COLOR[f.severity]
        p.add_run(f"{f.title}  (x{f.occurrence_count})").bold = True

        doc.add_paragraph(f.narrative)
        if f.escalation_reasons:
            esc_p = doc.add_paragraph()
            esc_p.add_run("Escalation: " + " ".join(f.escalation_reasons)).font.size = Pt(9)

    _add_heading(doc, "Prioritized Action List", level=1)
    table = doc.add_table(rows=1, cols=4)
    table.style = "Light Grid Accent 1"
    hdr = table.rows[0].cells
    hdr[0].text, hdr[1].text, hdr[2].text, hdr[3].text = "#", "Finding", "Severity", "Recommended Action"
    for a in report.action_list:
        row = table.add_row().cells
        row[0].text = str(a.priority_rank)
        row[1].text = a.finding_title
        row[2].text = _SEVERITY_LABEL[a.severity]
        row[3].text = a.recommended_action

    _add_heading(doc, "Methodology & Scope", level=1)
    doc.add_paragraph(
        f"Checklist: {report.checklist.name} ({report.checklist.key}). "
        f"{report.notes_processed} voice note(s) and {report.photos_processed} photo(s) processed; "
        f"{report.unmatched_notes} note(s)/photo(s) did not match any checklist item confidently and "
        "were excluded from findings rather than force-fit."
    )
    doc.add_paragraph(
        "This checklist is swappable -- the same pipeline can run a Lean 5S audit, a Theory of "
        "Constraints constraint walk, or a safety & compliance walk from the same field-note "
        "and photo inputs, depending on which YAML config is selected (or recommended)."
    )

    _add_heading(doc, "Appendix: Note-to-Finding Traceability", level=1)
    appx = doc.add_table(rows=1, cols=4)
    appx.style = "Light Grid Accent 1"
    ahdr = appx.rows[0].cells
    ahdr[0].text, ahdr[1].text, ahdr[2].text, ahdr[3].text = "Note ID", "Type", "Matched Finding", "Detail"
    for f in report.findings:
        for e in f.evidence:
            row = appx.add_row().cells
            row[0].text = e.note_id
            row[1].text = e.kind
            row[2].text = f.title
            row[3].text = e.detail[:140]

    doc.save(out_path)
    return out_path
