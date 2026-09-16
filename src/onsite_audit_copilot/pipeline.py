"""
Pipeline orchestrator: ingest raw field notes -> transcribe voice notes ->
extract entities -> tag against the active checklist -> caption and match
photos -> aggregate into findings -> build the action list -> synthesize
the executive summary -> render the report. Every stage is its own module
(see docs/architecture.md); this is the one function the CLI and the tests
call.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import checklist as checklist_mod
from .entity_extraction import extract_entities
from .findings import build_action_list, build_findings, synthesize_executive_summary
from .report import build_report
from .schema import FieldNoteInput, FindingsReport, PhotoMatch, TaggedNote
from .stt import get_transcript
from .vision import caption_photo


@dataclass
class VisitConfig:
    facility_name: str
    visit_area: str
    visit_date: str
    notes: list[FieldNoteInput]
    checklist_key: str | None = None            # explicit choice
    engagement_description: str | None = None   # used for recommend_checklist() if checklist_key is None
    synthetic_data_note: str | None = None


def run_audit(config: VisitConfig, out_dir: str | Path, render_pdf: bool = True) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if config.checklist_key:
        checklist = checklist_mod.load_checklist(config.checklist_key)
        recommendation_rationale, recommendation_method = None, None
    else:
        if not config.engagement_description:
            raise ValueError("Provide checklist_key or engagement_description.")
        checklist, recommendation_rationale, recommendation_method = checklist_mod.recommend_checklist(
            config.engagement_description
        )

    tagged_notes: list[TaggedNote] = []
    photo_matches: list[PhotoMatch] = []
    notes_processed = 0
    photos_processed = 0

    for note in config.notes:
        if note.kind == "voice":
            transcript = get_transcript(note.note_id, audio_path=note.audio_path, fixture_transcript=note.fixture_transcript)
            entities = extract_entities(transcript.text)
            tagged = checklist_mod.tag_note_to_checklist(transcript.text, entities, checklist, note.note_id)
            tagged_notes.append(tagged)
            notes_processed += 1
        elif note.kind == "photo":
            match = caption_photo(note.note_id, note.image_path, checklist, fixture_caption=note.fixture_caption)
            photo_matches.append(match)
            photos_processed += 1
        else:
            raise ValueError(f"Unknown note kind: {note.kind!r} (note {note.note_id})")

    findings, unmatched = build_findings(tagged_notes, photo_matches, checklist)
    action_list = build_action_list(findings)
    summary_text, summary_method = synthesize_executive_summary(
        config.facility_name, config.visit_area, checklist, findings
    )

    report = FindingsReport(
        facility_name=config.facility_name, visit_area=config.visit_area, visit_date=config.visit_date,
        checklist=checklist, executive_summary=summary_text, executive_summary_method=summary_method,
        findings=findings, action_list=action_list, notes_processed=notes_processed,
        photos_processed=photos_processed, unmatched_notes=unmatched,
        synthetic_data_note=config.synthetic_data_note,
    )

    report_path = build_report(report, out_dir / f"{_slug(config.visit_area)}_findings_report.docx")
    pdf_path = _convert_to_pdf(report_path) if render_pdf else None

    return {
        "report": report,
        "report_path": report_path,
        "pdf_path": pdf_path,
        "checklist_recommendation_rationale": recommendation_rationale,
        "checklist_recommendation_method": recommendation_method,
        "tagged_notes": tagged_notes,
        "photo_matches": photo_matches,
    }


def _slug(name: str) -> str:
    import re
    raw = "".join(c.lower() if c.isalnum() else "_" for c in name)
    return re.sub(r"_+", "_", raw).strip("_")


def _convert_to_pdf(docx_path: Path) -> Path | None:
    soffice = _find_soffice()
    if not soffice:
        return None
    try:
        subprocess.run(
            [soffice, "--headless", "--convert-to", "pdf", "--outdir", str(docx_path.parent), str(docx_path)],
            check=True, capture_output=True, timeout=90,
        )
    except Exception:
        return None
    pdf_path = docx_path.with_suffix(".pdf")
    return pdf_path if pdf_path.exists() else None


def _find_soffice() -> str | None:
    import shutil
    for candidate in ("soffice", "libreoffice"):
        path = shutil.which(candidate)
        if path:
            return path
    return None
