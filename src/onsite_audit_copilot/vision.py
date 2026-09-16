"""
Photo captioning and checklist matching. If ANTHROPIC_API_KEY is set, this
calls Claude's vision input directly on the photo (Claude is multimodal, so
no separate vision-model dependency is needed alongside the text LLM calls
in llm.py). Without a key, it falls back to a sidecar metadata file shipped
next to each synthetic photo -- see generate_synthetic_data.py -- which is
the honest stand-in for "read the caption the auditor typed when they took
the photo," not a fabricated AI output.

This module never sends a photo anywhere except the configured LLM
provider: there's no third-party image-recognition service being scraped
or otherwise abused here.
"""

from __future__ import annotations

from pathlib import Path

from .llm import caption_photo_llm, llm_available
from .schema import ChecklistConfig, PhotoMatch


def caption_photo(note_id: str, image_path: str | Path, checklist: ChecklistConfig, fixture_caption: dict | None = None) -> PhotoMatch:
    if llm_available():
        result = caption_photo_llm(
            image_path, [{"id": i.id, "category": i.category, "prompt": i.prompt} for i in checklist.items],
        )
        if result is not None:
            item_id = result.get("item_id")
            valid_item = item_id if item_id and checklist.get_item(item_id) else None
            return PhotoMatch(
                note_id=note_id, image_path=str(image_path),
                caption=result.get("caption", ""), checklist_item_id=valid_item,
                confidence="high" if valid_item else "low", method="llm_vision",
            )

    if fixture_caption is None:
        return PhotoMatch(
            note_id=note_id, image_path=str(image_path),
            caption="Photo captured on-site -- no caption available (vision LLM not configured and no sidecar metadata provided).",
            checklist_item_id=None, confidence="low", method="fixture_metadata",
        )

    item_id = fixture_caption.get("expected_checklist_item")
    valid_item = item_id if item_id and checklist.get_item(item_id) else None
    return PhotoMatch(
        note_id=note_id, image_path=str(image_path),
        caption=fixture_caption.get("scene_description", ""),
        checklist_item_id=valid_item,
        confidence="medium" if valid_item else "low",
        method="fixture_metadata",
    )
