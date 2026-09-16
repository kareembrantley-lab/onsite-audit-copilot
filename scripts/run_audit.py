#!/usr/bin/env python3
"""CLI for the on-site audit copilot. `--demo` regenerates the three
synthetic site visits and runs all of them, letting the checklist
recommender (not a hardcoded --checklist flag) pick the right lens for
each from its one-line engagement description -- the same "before running
it" stretch-goal behavior a real user gets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from onsite_audit_copilot.pipeline import VisitConfig, run_audit  # noqa: E402
from onsite_audit_copilot.schema import FieldNoteInput  # noqa: E402

DEMO_VISITS = ["final_assembly_5s", "packaging_ship_toc", "receiving_dock_safety"]


def _load_visit_config(slug: str) -> VisitConfig:
    visit_dir = ROOT / "data" / "synthetic" / slug
    meta = json.loads((visit_dir / "visit_config.json").read_text())
    raw_notes = json.loads((visit_dir / "notes.json").read_text())

    notes = []
    for n in raw_notes:
        if n["kind"] == "voice":
            notes.append(FieldNoteInput(note_id=n["note_id"], kind="voice", timestamp=n["timestamp"], fixture_transcript=n["fixture_transcript"]))
        elif n["kind"] == "photo":
            notes.append(FieldNoteInput(
                note_id=n["note_id"], kind="photo", timestamp=n["timestamp"],
                image_path=str(visit_dir / n["image_path"]), fixture_caption=n["fixture_caption"],
            ))
        else:
            raise ValueError(f"Unknown note kind in {slug}: {n['kind']!r}")

    return VisitConfig(
        facility_name=meta["facility_name"], visit_area=meta["visit_area"], visit_date=meta["visit_date"],
        notes=notes, engagement_description=meta["engagement_description"],
        synthetic_data_note=meta.get("synthetic_data_note"),
    )


def _print_summary(slug: str, result: dict) -> None:
    report = result["report"]
    print(f"\n=== {report.facility_name} -- {report.visit_area} ===")
    print(f"Checklist: {report.checklist.name} ({report.checklist.key})")
    if result["checklist_recommendation_rationale"]:
        print(f"  Recommended via {result['checklist_recommendation_method']}: {result['checklist_recommendation_rationale']}")
    print(f"Notes processed: {report.notes_processed}  Photos processed: {report.photos_processed}  Unmatched: {report.unmatched_notes}")
    for f in report.findings:
        print(f"  [{f.severity.upper():11s}] {f.title}  (x{f.occurrence_count})")
    print(f"Report: {result['report_path']}")
    if result["pdf_path"]:
        print(f"PDF:    {result['pdf_path']}")


def main():
    parser = argparse.ArgumentParser(description="On-site operational audit copilot.")
    parser.add_argument("--demo", action="store_true", help="Run all three synthetic demo visits.")
    parser.add_argument("--visit-dir", help="Path to a single visit's synthetic data directory (must contain visit_config.json and notes.json).")
    parser.add_argument("--checklist", help="Force a specific checklist key instead of letting the recommender pick one.")
    parser.add_argument("--out-dir", default=str(ROOT / "sample_output"), help="Output directory.")
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF rendering (requires LibreOffice).")
    args = parser.parse_args()

    if not args.demo and not args.visit_dir:
        parser.error("Provide --demo or --visit-dir.")

    if args.demo:
        for slug in DEMO_VISITS:
            config = _load_visit_config(slug)
            if args.checklist:
                config.checklist_key = args.checklist
            result = run_audit(config, out_dir=Path(args.out_dir) / slug, render_pdf=not args.no_pdf)
            _print_summary(slug, result)
        return

    visit_dir = Path(args.visit_dir)
    meta = json.loads((visit_dir / "visit_config.json").read_text())
    raw_notes = json.loads((visit_dir / "notes.json").read_text())
    notes = []
    for n in raw_notes:
        if n["kind"] == "voice":
            notes.append(FieldNoteInput(note_id=n["note_id"], kind="voice", timestamp=n["timestamp"], fixture_transcript=n["fixture_transcript"]))
        else:
            notes.append(FieldNoteInput(note_id=n["note_id"], kind="photo", timestamp=n["timestamp"], image_path=str(visit_dir / n["image_path"]), fixture_caption=n.get("fixture_caption")))
    config = VisitConfig(
        facility_name=meta["facility_name"], visit_area=meta["visit_area"], visit_date=meta["visit_date"],
        notes=notes, checklist_key=args.checklist, engagement_description=meta.get("engagement_description"),
        synthetic_data_note=meta.get("synthetic_data_note"),
    )
    result = run_audit(config, out_dir=Path(args.out_dir), render_pdf=not args.no_pdf)
    _print_summary(visit_dir.name, result)


if __name__ == "__main__":
    main()
